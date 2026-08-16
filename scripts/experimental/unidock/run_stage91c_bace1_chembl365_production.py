"""Run the frozen BACE1 ChEMBL-365 x 34 x 3 Uni-Dock matrix."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections import Counter
from pathlib import Path

from scripts.experimental.unidock import run_stage09_mk14_train696_production as common


FROZEN_SEEDS = (("seed0", 20260801), ("seed1", 20260802), ("seed2", 20260803))
FROZEN_PROFILE = ("enhanced", 1024, 80)


def verify_implementation(
    root: Path, config: dict[str, object], key: str, expected_path: Path
) -> None:
    descriptor = dict(config["implementation"])[key]
    path = common.rooted_path(root, str(descriptor["path"]))
    if path.resolve() != expected_path.resolve():
        raise ValueError(f"Stage91c implementation path differs: {key}")
    if common.file_sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"Stage91c implementation hash differs: {key}")


def validate_config(config: dict[str, object]) -> None:
    required = {
        "schema_version", "experiment_id", "purpose", "implementation",
        "data_boundary", "inputs", "expected", "unidock", "execution",
        "outputs", "decision_boundary",
    }
    if set(config) != required:
        raise ValueError("Stage91c config keys differ")
    boundary = dict(config["data_boundary"])
    if int(boundary["confirmation_rows_permitted"]) != 0 or int(
        boundary["locked_test_rows_permitted"]
    ) != 0:
        raise ValueError("Stage91c crossed the frozen data boundary")
    seeds = tuple(
        (str(row["seed_id"]), int(row["base_seed"]))
        for row in dict(config["inputs"])["seeds"]
    )
    if seeds != FROZEN_SEEDS:
        raise ValueError("Stage91c seed ledger differs")
    expected = dict(config["expected"])
    fixed_counts = {
        "receptor_count": 34,
        "ligand_count": 365,
        "seed_count": 3,
        "batch_count": 102,
        "pair_count": 37230,
        "confirmation_rows": 0,
        "locked_test_rows": 0,
    }
    for key, value in fixed_counts.items():
        if int(expected[key]) != value:
            raise ValueError(f"Stage91c expected count differs: {key}")
    protocol = dict(config["unidock"])
    profile = (
        str(protocol["profile_id"]),
        int(protocol["exhaustiveness"]),
        int(protocol["max_step"]),
    )
    if profile != FROZEN_PROFILE:
        raise ValueError("Stage91c frozen Uni-Dock profile differs")
    if str(protocol["required_package_version"]) != "1.1.3":
        raise ValueError("Stage91c Uni-Dock version differs")
    if int(protocol["num_modes"]) != 1:
        raise ValueError("Stage91c must retain exactly one pose")


def validate_inputs(
    root: Path, config: dict[str, object]
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, object]]:
    inputs = dict(config["inputs"])
    expected = dict(config["expected"])
    boundary = dict(config["data_boundary"])
    receptor_path = common.verified_path(root, dict(inputs["receptor_manifest"]))
    ligand_path = common.verified_path(root, dict(inputs["ligand_manifest"]))
    coverage_path = common.verified_path(root, dict(inputs["structural_coverage_gate"]))
    stage91_path = common.verified_path(root, dict(inputs["stage91_preregistration"]))
    docking_prereg_path = common.verified_path(root, dict(inputs["docking_preregistration"]))
    audit_path = common.verified_path(root, dict(inputs["input_preparation_audit"]))
    preparation_path = common.verified_path(root, dict(inputs["ligand_preparation_summary"]))
    profile_path = common.verified_path(root, dict(inputs["profile_freeze_result"]))

    coverage = common.read_json(coverage_path)
    stage91 = common.read_json(stage91_path)
    docking_prereg = common.read_json(docking_prereg_path)
    input_audit = common.read_json(audit_path)
    preparation = common.read_json(preparation_path)
    profile = common.read_json(profile_path)
    if coverage.get("status") != "stage41d_conditional_go_new_posthoc_development_route":
        raise ValueError("the 34-receptor structural coverage gate did not pass")
    if stage91.get("status") != "stage91_bace1_group_robust_rescue_preregistered":
        raise ValueError("the Stage91 objective was not preregistered")
    if docking_prereg.get("status") != "stage91c_development_docking_preregistered":
        raise ValueError("Stage91c development docking was not preregistered")
    authorization = dict(docking_prereg["authorization"])
    if not authorization.get("development_docking_authorized") or authorization.get(
        "confirmation_or_test_docking_authorized"
    ):
        raise ValueError("Stage91c docking authorization differs")
    if input_audit.get("status") != "stage91b_bace1_chembl365_input_independent_audit_ok":
        raise ValueError("the independent Stage91b input audit did not pass")
    if not input_audit.get("development_docking_release"):
        raise ValueError("the Stage91b audit did not release development docking")
    if preparation.get("status") != "stage91b_bace1_chembl365_unidock_inputs_ok":
        raise ValueError("the Stage91b ligand preparation did not pass")
    if profile.get("status") != "unidock_profile_frozen_train_only" or profile.get(
        "selected_profile_id"
    ) != "enhanced":
        raise ValueError("the frozen Uni-Dock profile differs")

    receptors = common.read_csv(receptor_path)
    ligands = common.read_csv(ligand_path)
    receptor_ids = [row["conformer_id"] for row in receptors]
    if receptor_ids != [str(value) for value in expected["receptor_ids"]]:
        raise ValueError("Stage91c receptor order differs")
    if receptor_ids != list(coverage["passing_receptor_ids"]):
        raise ValueError("Stage91c receptors differ from the qualified pool")
    if len(receptors) != 34 or any(row["status"] != "ok" for row in receptors):
        raise ValueError("Stage91c receptor manifest differs")
    if len(ligands) != 365 or len({row["ligand_id"] for row in ligands}) != 365:
        raise ValueError("Stage91c ligand grid differs")
    labels = Counter(row["label"] for row in ligands)
    expected_labels = Counter(
        {key: int(value) for key, value in dict(expected["potency_label_counts"]).items()}
    )
    if labels != expected_labels:
        raise ValueError("Stage91c potency-label counts differ")
    if {row["role"] for row in ligands} != {boundary["allowed_role"]}:
        raise ValueError("Stage91c exposed a nondevelopment role")
    if {row["split"] for row in ligands} != {boundary["allowed_split"]}:
        raise ValueError("Stage91c exposed a nondevelopment split")
    if {row["selection_role"] for row in ligands} != {
        boundary["allowed_selection_role"]
    }:
        raise ValueError("Stage91c ligand selection role differs")
    if any(row["pdbqt_status"] != "ok" for row in ligands):
        raise ValueError("Stage91c ligand manifest contains a failed PDBQT")

    for rows, path_column, hash_column, id_column in (
        (receptors, "receptor_pdbqt", "receptor_pdbqt_sha256", "conformer_id"),
        (ligands, "pdbqt_path", "pdbqt_sha256", "ligand_id"),
    ):
        for row in rows:
            path = common.rooted_path(root, row[path_column])
            if not path.is_file() or common.file_sha256(path) != row[hash_column].upper():
                raise ValueError(f"Stage91c prepared input differs: {row[id_column]}")
    pseudoatom_ids = [
        row["ligand_id"]
        for row in ligands
        if common.macrocycle_closure_atom_types(
            common.rooted_path(root, row["pdbqt_path"])
        )
    ]
    if pseudoatom_ids:
        raise ValueError(f"Stage91c ligands retain closure pseudoatoms: {pseudoatom_ids}")
    variants = Counter(row["preparation_variant"] for row in ligands)
    if variants != Counter(
        {key: int(value) for key, value in preparation["preparation_variant_counts"].items()}
    ):
        raise ValueError("Stage91c preparation variants differ")
    return receptors, ligands, {
        "status": "audit_only_ok",
        "target_id": "BACE1",
        "experiment_class": "prospective_objective_frozen_development",
        "receptor_count": len(receptors),
        "receptor_ids": receptor_ids,
        "ligand_count": len(ligands),
        "potency_label_counts": dict(sorted(labels.items())),
        "preparation_variant_counts": dict(sorted(variants.items())),
        "macrocycle_closure_pseudoatom_ligand_count": 0,
        "seed_count": len(FROZEN_SEEDS),
        "expected_batch_count": 102,
        "expected_pair_count": 37230,
        "confirmation_rows": 0,
        "locked_test_rows": 0,
    }


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
    config_sha256 = common.file_sha256(config_path)
    rows, summaries, missing = common.collect_batches(
        root, config, receptors, ligands, config_sha256
    )
    progress_path = common.rooted_path(root, str(outputs["progress_json"]))
    progress = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage91c_production_complete" if not missing else "stage91c_partial_ok",
        "selected_seed_ids": selected_seed_ids,
        "selected_receptor_ids": selected_receptor_ids,
        "completed_batch_count": len(summaries),
        "missing_batch_count": len(missing),
        "missing_batches": missing,
        "completed_pair_count": len(rows),
        "expected_batch_count": 102,
        "expected_pair_count": 37230,
        "executed_batches_this_invocation": executed_batches,
        "resumed_batches_this_invocation": resumed_batches,
        "current_invocation_elapsed_seconds": invocation_elapsed,
        "confirmation_rows_read": 0,
        "locked_test_rows_read": 0,
    }
    common.write_json(progress_path, progress)
    if missing:
        print(json.dumps(progress, indent=2, sort_keys=True))
        return progress

    if len(rows) != 37230:
        raise ValueError("complete Stage91c pair count differs")
    unique = {
        (str(row["seed_id"]), str(row["receptor_id"]), str(row["ligand_id"]))
        for row in rows
    }
    if len(unique) != 37230:
        raise ValueError("complete Stage91c keys are not unique")
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
            raise ValueError(f"invalid Stage91c score: {row['ligand_id']}")

    scores_path = common.rooted_path(root, str(outputs["scores_csv"]))
    batches_path = common.rooted_path(root, str(outputs["batch_runs_csv"]))
    median_path = common.rooted_path(root, str(outputs["median_matrix_csv"]))
    minimum_path = common.rooted_path(root, str(outputs["minimum_matrix_csv"]))
    summary_path = common.rooted_path(root, str(outputs["summary_json"]))
    common.write_csv(scores_path, rows)
    batch_rows = [
        {
            "seed_id": summary["seed_id"],
            "base_seed": summary["base_seed"],
            "receptor_id": summary["receptor_id"],
            "ligand_count": summary["ligand_count"],
            "elapsed_seconds": summary["elapsed_seconds"],
            "score_minimum": summary["score_minimum"],
            "score_maximum": summary["score_maximum"],
            "known_warning_event_count": summary["warning_adjudication"]["known_warning_event_count"],
            "unresolved_warning_event_count": summary["warning_adjudication"]["unresolved_warning_event_count"],
            "pose_integrity_failure_count": summary["pose_integrity_audit"]["failure_count"],
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
    common.write_csv(batches_path, batch_rows)
    common.write_csv(median_path, common.matrix_rows(rows, ligands, receptor_ids, "median"))
    common.write_csv(minimum_path, common.matrix_rows(rows, ligands, receptor_ids, "minimum"))
    batch_elapsed = sum(float(row["elapsed_seconds"]) for row in batch_rows)
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage91c_bace1_chembl365_unidock_matrix_ok",
        "operation": "preregistered development-only Uni-Dock production score generation",
        "experiment_class": "prospective_objective_frozen_development",
        "config": {"path": common.relative_path(root, config_path), "sha256": config_sha256},
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
        "batch_elapsed_seconds": batch_elapsed,
        "pairs_per_batch_second": len(rows) / batch_elapsed,
        "executed_batches_this_invocation": executed_batches,
        "resumed_batches_this_invocation": resumed_batches,
        "current_invocation_elapsed_seconds": invocation_elapsed,
        "aggregation": {
            "primary": "median across the three paired seeds",
            "sensitivity": "minimum across the three paired seeds",
            "score_direction": "more negative is more favorable",
        },
        "data_boundary": {"confirmation_rows_read": 0, "locked_test_rows_read": 0},
        "outputs": {
            "scores_csv": common.output_descriptor(root, scores_path),
            "batch_runs_csv": common.output_descriptor(root, batches_path),
            "median_matrix_csv": common.output_descriptor(root, median_path),
            "minimum_matrix_csv": common.output_descriptor(root, minimum_path),
            "progress_json": common.output_descriptor(root, progress_path),
        },
        "next_gate": "independent Stage91c matrix audit before any QUBO or comparator analysis",
        "interpretation_boundary": config["decision_boundary"],
    }
    common.write_json(summary_path, result)
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
    config = common.read_json(config_path)
    validate_config(config)
    implementation_paths = {
        "runner": Path(__file__),
        "independent_auditor": Path(__file__).with_name(
            "audit_stage91c_bace1_chembl365_production.py"
        ),
        "unidock_batch_helper": Path(__file__).with_name("run_unidock_gpu_equivalence.py"),
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
    selected_seeds = common.selected_records(seeds, "seed_id", seed_ids)
    selected_receptors = common.selected_records(
        [dict(row) for row in receptors], "conformer_id", receptor_ids
    )
    selected_seed_ids = [str(row["seed_id"]) for row in selected_seeds]
    selected_receptor_ids = [str(row["conformer_id"]) for row in selected_receptors]
    if audit_only:
        result = {
            "schema_version": "1.0",
            "experiment_id": config["experiment_id"],
            "config": {
                "path": common.relative_path(root, config_path),
                "sha256": common.file_sha256(config_path),
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
        executable_info = common.executable_evidence(
            unidock or str(protocol["executable"]),
            str(protocol["required_package_version"]),
        )
        run_directory = common.rooted_path(root, str(config["outputs"]["run_directory"]))
        config_sha256 = common.file_sha256(config_path)
        ligand_ids = {row["ligand_id"] for row in ligands}
        for seed in selected_seeds:
            seed_id = str(seed["seed_id"])
            base_seed = int(seed["base_seed"])
            for receptor in selected_receptors:
                receptor_id = str(receptor["conformer_id"])
                paths = common.batch_paths(run_directory, seed_id, receptor_id)
                paths["directory"].mkdir(parents=True, exist_ok=True)
                signature = common.protocol_signature(
                    config_sha256, seed_id, base_seed, receptor, ligands, protocol
                )
                value = (
                    common.checkpoint(root, paths, signature, ligand_ids)
                    if resume
                    else None
                )
                if value is not None:
                    resumed_batches += 1
                    print(f"resume ok: {seed_id}/{receptor_id}", flush=True)
                    continue
                print(f"running: {seed_id}/{receptor_id}", flush=True)
                rows, summary = common.run_batch(
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
                rows, pose_audit = common.audit_batch_poses(root, ligands, rows)
                warning = common.classify_warning_log(paths["log"], pose_audit)
                summary["pose_integrity_audit"] = pose_audit
                summary["warning_adjudication"] = warning
                summary["status"] = (
                    "ok"
                    if int(pose_audit["failure_count"]) == 0
                    and int(warning["unresolved_warning_event_count"]) == 0
                    else "technical_integrity_failed"
                )
                common.write_csv(paths["scores"], rows)
                summary["scores_sha256"] = common.file_sha256(paths["scores"])
                common.write_json(paths["summary"], summary)
                if summary["status"] != "ok":
                    raise ValueError(
                        f"Stage91c technical gate failed: {seed_id}/{receptor_id}"
                    )
                executed_batches += 1
                print(
                    f"completed: {seed_id}/{receptor_id} in "
                    f"{float(summary['elapsed_seconds']):.3f} s",
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
        "stage91c_partial_ok",
        "stage91c_bace1_chembl365_unidock_matrix_ok",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
