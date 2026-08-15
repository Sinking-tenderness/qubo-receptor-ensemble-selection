"""Run the frozen three-seed PPARD cognate-redocking gate."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from scripts.experimental.unidock import run_stage14d_fa10_cognate_redocking as common


EXPECTED_SEEDS = (("seed0", 20260801), ("seed1", 20260802), ("seed2", 20260803))


def verify_implementation(root: Path, config: dict[str, object], key: str, expected: Path) -> None:
    descriptor = dict(config["implementation"])[key]
    path = common.rooted_path(root, str(descriptor["path"]))
    if path.resolve() != expected.resolve():
        raise ValueError(f"Stage57 implementation path differs: {key}")
    if common.file_sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"Stage57 implementation hash differs: {key}")


def runtime_path(root: Path, value: str) -> Path:
    path = common.rooted_path(root, value)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def validate_inputs(
    root: Path, config: dict[str, object]
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, object]]:
    implementations = {
        "runner": Path(__file__),
        "batch_helper": Path(__file__).with_name("run_unidock_gpu_equivalence.py"),
        "pose_audit_helper": Path(__file__).with_name(
            "run_stage07b_unidock_enhanced_confirmation.py"
        ),
        "warning_helper": Path(__file__).with_name(
            "run_stage07c_unidock_warning_adjudication.py"
        ),
        "rmsd_helper": root / "scripts/evaluate_redocking_rmsd.py",
    }
    for key, path in implementations.items():
        verify_implementation(root, config, key, path)
    if any(int(value) != 0 for value in dict(config["data_boundary"]).values()):
        raise ValueError("Stage57 redocking crossed a frozen data boundary")

    inputs = dict(config["inputs"])
    prep_config = common.verified(root, dict(inputs["preparation_config"]))
    frozen_manifest_path = common.verified(root, dict(inputs["frozen_receptor_manifest"]))
    preparation_path = runtime_path(root, str(inputs["preparation_summary_path"]))
    receptor_path = runtime_path(root, str(inputs["receptor_manifest_path"]))
    case_path = runtime_path(root, str(inputs["case_manifest_path"]))
    box_path = runtime_path(root, str(inputs["common_box_path"]))
    preparation = common.read_json(preparation_path)
    if preparation.get("status") not in {
        "stage57_ppard_redocking_inputs_ok",
        "stage57_ppard_redocking_inputs_partial_gate_ready",
    }:
        raise ValueError("Stage57 PPARD input preparation did not pass")
    if preparation.get("technical_gate_ready") is not True:
        raise ValueError("Stage57 PPARD technical input gate is not ready")
    if preparation["config"]["sha256"] != common.file_sha256(prep_config):
        raise ValueError("Stage57 preparation config identity differs")
    if any(int(value) != 0 for value in dict(preparation["data_boundary"]).values()):
        raise ValueError("Stage57 preparation crossed a protected boundary")
    runtime_outputs = {
        "receptor_manifest_csv": receptor_path,
        "redocking_case_manifest_csv": case_path,
        "common_box_json": box_path,
    }
    for key, path in runtime_outputs.items():
        source = dict(preparation["outputs"])[key]
        if common.rooted_path(root, str(source["path"])).resolve() != path.resolve():
            raise ValueError(f"Stage57 preparation output path differs: {key}")
        if common.file_sha256(path) != str(source["sha256"]).upper():
            raise ValueError(f"Stage57 preparation output identity differs: {key}")

    frozen = common.read_csv(frozen_manifest_path)
    receptors = common.read_csv(receptor_path)
    cases = common.read_csv(case_path)
    common_box = common.read_json(box_path)
    frozen_ids = [row["conformer_id"] for row in frozen]
    receptor_ids = [row["conformer_id"] for row in receptors]
    expected = dict(config["expected"])
    if len(frozen) != int(expected["frozen_receptor_count"]):
        raise ValueError("Stage57 frozen receptor denominator differs")
    expected_prepared_order = [value for value in frozen_ids if value in set(receptor_ids)]
    if receptor_ids != expected_prepared_order:
        raise ValueError("Stage57 prepared receptor order differs from the frozen pool")
    if len(receptors) < int(expected["minimum_prepared_receptor_count"]):
        raise ValueError("Stage57 prepared receptor count is below the frozen minimum")
    if [row["conformer_id"] for row in cases] != receptor_ids:
        raise ValueError("Stage57 receptor and cognate-case order differs")
    if len({row["case_id"] for row in cases}) != len(cases):
        raise ValueError("Stage57 case IDs are not unique")
    if any(row["status"] != "ok" for row in receptors + cases):
        raise ValueError("Stage57 prepared manifest contains a failed row")
    failures = [dict(value) for value in preparation["failed_cases"]]
    failure_ids = [str(value["conformer_id"]) for value in failures]
    if len(receptors) + len(failures) != len(frozen):
        raise ValueError("Stage57 prepared/failure accounting differs")
    if set(receptor_ids).intersection(failure_ids):
        raise ValueError("Stage57 receptor is both prepared and failed")
    if set(receptor_ids).union(failure_ids) != set(frozen_ids):
        raise ValueError("Stage57 preparation does not cover the frozen pool")
    for row in receptors:
        path = common.rooted_path(root, row["receptor_pdbqt"])
        if not path.is_file() or common.file_sha256(path) != row["receptor_pdbqt_sha256"].upper():
            raise ValueError(f"Stage57 receptor identity differs: {row['conformer_id']}")
    for row in cases:
        for path_key, hash_key in (
            ("ligand_pdbqt", "ligand_pdbqt_sha256"),
            ("reference_sdf", "reference_sdf_sha256"),
        ):
            path = common.rooted_path(root, row[path_key])
            if not path.is_file() or common.file_sha256(path) != row[hash_key].upper():
                raise ValueError(f"Stage57 case identity differs: {row['case_id']}")
        if common.macrocycle_closure_atom_types(
            common.rooted_path(root, row["ligand_pdbqt"])
        ):
            raise ValueError(f"Stage57 ligand retains closure pseudoatoms: {row['case_id']}")
    seeds = tuple(
        (str(value["seed_id"]), int(value["base_seed"]))
        for value in inputs["seeds"]
    )
    if seeds != EXPECTED_SEEDS:
        raise ValueError("Stage57 seed ledger differs")
    if len(receptors) * len(seeds) > int(expected["maximum_redocking_pair_count"]):
        raise ValueError("Stage57 redocking grid exceeds the frozen maximum")
    common.validate_protocol(dict(config["unidock"]), common_box)
    return receptors, cases, {
        "status": "audit_only_ok",
        "target_id": "PPARD",
        "receptor_count": len(receptors),
        "receptor_ids": receptor_ids,
        "case_count": len(cases),
        "seed_count": len(seeds),
        "expected_redocking_pair_count": len(receptors) * len(seeds),
        "frozen_receptor_count": len(frozen),
        "technical_preparation_failure_count": len(failures),
        "technical_preparation_failures": failures,
        "common_box": common_box,
        "ligand_labels_read": 0,
        "benchmark_docking_scores_read": 0,
        "fresh_validation_rows_read": 0,
        "test_rows_read": 0,
    }


def summarize_gate(
    rows: list[dict[str, object]], receptor_ids: list[str], threshold: float, minimum_successes: int
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for receptor_id in receptor_ids:
        selected = [row for row in rows if row["conformer_id"] == receptor_id]
        if len(selected) != len(EXPECTED_SEEDS):
            raise ValueError(f"Stage57 receptor result count differs: {receptor_id}")
        rmsds = [float(row["top_ranked_rmsd_angstrom"]) for row in selected]
        successes = sum(common.truth(row["top_ranked_pose_success"]) for row in selected)
        output.append(
            {
                "conformer_id": receptor_id,
                "seed_count": len(selected),
                "successful_seed_count": successes,
                "median_top_ranked_rmsd_angstrom": statistics.median(rmsds),
                "maximum_top_ranked_rmsd_angstrom": max(rmsds),
                "gate_pass": successes >= minimum_successes
                and statistics.median(rmsds) <= threshold,
            }
        )
    return output


def run(
    config_path: Path,
    root: Path,
    unidock: str | None,
    audit_only: bool,
    resume: bool,
) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = common.read_json(config_path)
    receptors, cases, input_audit = validate_inputs(root, config)
    config_hash = common.file_sha256(config_path)
    if audit_only:
        result = {
            "schema_version": "1.0",
            "experiment_id": config["experiment_id"],
            "config": {"path": common.relative_path(root, config_path), "sha256": config_hash},
            **input_audit,
            "operation": "input audit only; no GPU docking was started",
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    runtime = common.runtime_evidence(config)
    protocol = dict(config["unidock"])
    executable = common.executable_evidence(
        unidock or str(protocol["executable"]),
        str(protocol["required_package_version"]),
    )
    run_root = common.rooted_path(root, str(dict(config["outputs"])["run_directory"]))
    run_root.mkdir(parents=True, exist_ok=True)
    receptor_by_id = {row["conformer_id"]: row for row in receptors}
    threshold = float(dict(config["redocking_gate"])["maximum_rmsd_angstrom"])
    all_rows: list[dict[str, object]] = []
    executed = 0
    resumed = 0
    for case in cases:
        receptor = receptor_by_id[case["conformer_id"]]
        ligand = {
            "ligand_id": case["case_id"],
            "label": "cognate_control",
            "selection_role": "technical_redocking",
            "pdbqt_path": case["ligand_pdbqt"],
            "pdbqt_sha256": case["ligand_pdbqt_sha256"],
        }
        for seed in dict(config["inputs"])["seeds"]:
            seed_id = str(seed["seed_id"])
            base_seed = int(seed["base_seed"])
            batch = common.batch_paths(run_root, seed_id, receptor["conformer_id"])
            batch["directory"].mkdir(parents=True, exist_ok=True)
            signature = common.protocol_signature(
                config_hash, seed_id, base_seed, receptor, [ligand], protocol
            )
            cached = (
                common.validate_checkpoint(root, batch, signature, {case["case_id"]})
                if resume
                else None
            )
            if cached is not None:
                batch_rows, batch_summary = cached
                if "warning_adjudication" not in batch_summary or not all(
                    "pose_integrity_status" in row for row in batch_rows
                ):
                    cached = None
            if cached is None:
                print(f"running: {seed_id}/{receptor['conformer_id']}", flush=True)
                batch_rows, batch_summary = common.run_batch(
                    root,
                    batch,
                    str(executable["resolved_executable"]),
                    receptor,
                    [ligand],
                    protocol,
                    seed_id,
                    base_seed,
                    signature,
                )
                for row in batch_rows:
                    row["target_id"] = "PPARD"
                batch_rows, pose_audit = common.audit_batch_poses(root, [ligand], batch_rows)
                warning = common.classify_warning_log(batch["log"], pose_audit)
                batch_summary["pose_integrity"] = pose_audit
                batch_summary["warning_adjudication"] = warning
                common.write_csv(batch["scores"], batch_rows)
                batch_summary["scores_sha256"] = common.file_sha256(batch["scores"])
                common.write_json(batch["summary"], batch_summary)
                executed += 1
            else:
                batch_rows, batch_summary = cached
                resumed += 1
                print(f"resume ok: {seed_id}/{receptor['conformer_id']}", flush=True)
            output_pose = common.rooted_path(root, str(batch_rows[0]["output_pose_path"]))
            rmsd_dir = batch["directory"] / "rmsd"
            rmsd_summary_path = rmsd_dir / "summary.json"
            pose_table = rmsd_dir / "poses.csv"
            common.run_checked(
                [
                    sys.executable,
                    str(root / "scripts/evaluate_redocking_rmsd.py"),
                    "--case-id",
                    f"{case['case_id']}__{seed_id}",
                    "--reference-sdf",
                    str(common.rooted_path(root, case["reference_sdf"])),
                    "--docked-pdbqt",
                    str(output_pose),
                    "--pose-table-output",
                    str(pose_table),
                    "--summary-output",
                    str(rmsd_summary_path),
                    "--success-threshold",
                    str(threshold),
                ],
                f"PPARD redocking RMSD evaluation for {case['case_id']} {seed_id}",
            )
            rmsd = common.read_json(rmsd_summary_path)
            warning = dict(batch_summary["warning_adjudication"])
            pose_audit = dict(batch_summary["pose_integrity"])
            all_rows.append(
                {
                    "target_id": "PPARD",
                    "case_id": case["case_id"],
                    "conformer_id": receptor["conformer_id"],
                    "seed_id": seed_id,
                    "base_seed": base_seed,
                    "reference_ligand": case["selected_ligand_resname"],
                    "reference_sdf": case["reference_sdf"],
                    "reference_sdf_sha256": case["reference_sdf_sha256"],
                    "docked_pdbqt": common.relative_path(root, output_pose),
                    "docked_pdbqt_sha256": common.file_sha256(output_pose),
                    "top_ranked_affinity_kcal_per_mol": rmsd["top_ranked_affinity_kcal_per_mol"],
                    "top_ranked_rmsd_angstrom": rmsd["top_ranked_rmsd_angstrom"],
                    "top_ranked_pose_success": rmsd["top_ranked_pose_success"],
                    "pose_integrity_failure_count": pose_audit["failure_count"],
                    "known_warning_event_count": warning["known_warning_event_count"],
                    "unresolved_warning_event_count": warning["unresolved_warning_event_count"],
                    "batch_summary": common.relative_path(root, batch["summary"]),
                    "batch_summary_sha256": common.file_sha256(batch["summary"]),
                    "evaluation_summary": common.relative_path(root, rmsd_summary_path),
                    "evaluation_summary_sha256": common.file_sha256(rmsd_summary_path),
                    "status": "ok",
                }
            )

    expected_pairs = int(input_audit["expected_redocking_pair_count"])
    if len(all_rows) != expected_pairs or len(
        {(row["conformer_id"], row["seed_id"]) for row in all_rows}
    ) != expected_pairs:
        raise ValueError("Stage57 complete redocking grid differs")
    gate = dict(config["redocking_gate"])
    gate_rows = summarize_gate(
        all_rows,
        input_audit["receptor_ids"],
        threshold,
        int(gate["minimum_successful_seeds_per_receptor"]),
    )
    for failure in input_audit["technical_preparation_failures"]:
        gate_rows.append(
            {
                "conformer_id": failure["conformer_id"],
                "seed_count": 0,
                "successful_seed_count": 0,
                "median_top_ranked_rmsd_angstrom": "",
                "maximum_top_ranked_rmsd_angstrom": "",
                "gate_pass": False,
                "failure_stage": "stage57_input_preparation",
                "failure_reason": failure["error"],
            }
        )
    unresolved = sum(int(row["unresolved_warning_event_count"]) for row in all_rows)
    pose_failures = sum(int(row["pose_integrity_failure_count"]) for row in all_rows)
    passed = sum(common.truth(row["gate_pass"]) for row in gate_rows)
    minimum_passing = int(gate["minimum_passing_receptor_count"])
    status = (
        "stage57_ppard_cognate_redocking_gate_ok"
        if passed >= minimum_passing and unresolved == 0 and pose_failures == 0
        else "stage57_ppard_cognate_redocking_gate_failed"
    )
    outputs = dict(config["outputs"])
    result_path = common.rooted_path(root, str(outputs["redocking_results_csv"]))
    gate_path = common.rooted_path(root, str(outputs["receptor_gate_results_csv"]))
    summary_path = common.rooted_path(root, str(outputs["summary_json"]))
    common.write_csv(result_path, all_rows)
    common.write_csv(gate_path, gate_rows)
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": status,
        "config": {"path": common.relative_path(root, config_path), "sha256": config_hash},
        "runtime": runtime,
        "unidock_executable": executable,
        "frozen_protocol": protocol,
        "input_audit": input_audit,
        "redocking_pair_count": len(all_rows),
        "executed_batches_this_invocation": executed,
        "resumed_batches_this_invocation": resumed,
        "receptor_gate_results": gate_rows,
        "passed_receptor_count": passed,
        "frozen_receptor_count": input_audit["frozen_receptor_count"],
        "technical_preparation_failure_count": input_audit[
            "technical_preparation_failure_count"
        ],
        "failed_receptor_count": int(input_audit["frozen_receptor_count"]) - passed,
        "minimum_passing_receptor_count": minimum_passing,
        "technical_gate_pass": status.endswith("_ok"),
        "known_warning_event_count": sum(
            int(row["known_warning_event_count"]) for row in all_rows
        ),
        "unresolved_warning_event_count": unresolved,
        "pose_integrity_failure_count": pose_failures,
        "data_boundary": {
            "ligand_labels_read": 0,
            "benchmark_docking_scores_read": 0,
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
        },
        "outputs": {
            "redocking_results_csv": {
                "path": common.relative_path(root, result_path),
                "sha256": common.file_sha256(result_path),
            },
            "receptor_gate_results_csv": {
                "path": common.relative_path(root, gate_path),
                "sha256": common.file_sha256(gate_path),
            },
        },
        "next_gate": (
            "dock the frozen 96-ligand PPARD development pilot against every passing receptor"
            if status.endswith("_ok")
            else "stop PPARD before pilot production and preserve the failed technical gate"
        ),
        "decision_boundary": config["decision_boundary"],
    }
    common.write_json(summary_path, summary)
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
    run(args.config, args.root, args.unidock, args.audit_only, args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
