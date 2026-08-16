"""Run the frozen PPARA large-pool, three-seed cognate-redocking gate."""

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
        raise ValueError(f"Stage51 implementation path differs: {key}")
    if common.file_sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"Stage51 implementation hash differs: {key}")


def validate_inputs(
    root: Path, config: dict[str, object]
) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, object]]:
    implementations = {
        "runner": Path(__file__),
        "batch_helper": Path(__file__).with_name("run_unidock_gpu_equivalence.py"),
        "pose_audit_helper": Path(__file__).with_name("run_stage07b_unidock_enhanced_confirmation.py"),
        "warning_helper": Path(__file__).with_name("run_stage07c_unidock_warning_adjudication.py"),
        "rmsd_helper": root / "scripts/evaluate_redocking_rmsd.py",
    }
    for key, path in implementations.items():
        verify_implementation(root, config, key, path)
    if any(int(value) != 0 for value in dict(config["data_boundary"]).values()):
        raise ValueError("Stage51 crossed a frozen data boundary")
    inputs = dict(config["inputs"])
    paths = {
        key: common.verified(root, dict(value))
        for key, value in inputs.items()
        if isinstance(value, dict)
    }
    preparation = common.read_json(paths["preparation_summary"])
    preparation_audit = common.read_json(paths["preparation_audit"])
    common_box = common.read_json(paths["common_box"])
    if preparation.get("status") != "stage50_ppara_large_pool_redocking_inputs_partial_gate_ready":
        raise ValueError("Stage50 PPARA input preparation did not pass its partial gate")
    if preparation.get("technical_gate_ready") is not True:
        raise ValueError("Stage50 PPARA technical gate is not ready")
    if preparation_audit.get("status") != "stage50_ppara_large_pool_inputs_independent_audit_ok":
        raise ValueError("Stage50 independent input audit did not pass")
    if preparation_audit.get("cognate_redocking_authorized") is not True:
        raise ValueError("Stage50 independent audit did not authorize redocking")
    if any(int(value) != 0 for value in dict(preparation["data_boundary"]).values()):
        raise ValueError("Stage50 preparation crossed a data boundary")
    for key in ("receptor_manifest_csv", "redocking_case_manifest_csv", "common_box_json"):
        source = dict(preparation["outputs"])[key]
        if common.file_sha256(common.rooted_path(root, str(source["path"]))) != str(source["sha256"]).upper():
            raise ValueError(f"Stage50 output identity differs: {key}")

    receptors = common.read_csv(paths["receptor_manifest"])
    cases = common.read_csv(paths["case_manifest"])
    expected = dict(config["expected"])
    receptor_ids = [row["conformer_id"] for row in receptors]
    if receptor_ids != [str(value) for value in expected["receptor_ids"]]:
        raise ValueError("Stage51 receptor order differs")
    if len(receptors) != int(expected["receptor_count"]) or len(cases) != int(expected["case_count"]):
        raise ValueError("Stage51 receptor or case count differs")
    if [row["conformer_id"] for row in cases] != receptor_ids:
        raise ValueError("Stage51 receptor and cognate-case order differs")
    if len({row["case_id"] for row in cases}) != len(cases):
        raise ValueError("Stage51 case IDs are not unique")
    if any(row["status"] != "ok" for row in receptors + cases):
        raise ValueError("Stage51 input manifest contains a failed row")
    for row in receptors:
        path = common.rooted_path(root, row["receptor_pdbqt"])
        if not path.is_file() or common.file_sha256(path) != row["receptor_pdbqt_sha256"].upper():
            raise ValueError(f"Stage51 receptor identity differs: {row['conformer_id']}")
    for row in cases:
        for path_key, hash_key in (("ligand_pdbqt", "ligand_pdbqt_sha256"), ("reference_sdf", "reference_sdf_sha256")):
            path = common.rooted_path(root, row[path_key])
            if not path.is_file() or common.file_sha256(path) != row[hash_key].upper():
                raise ValueError(f"Stage51 case identity differs: {row['case_id']}")
        if common.macrocycle_closure_atom_types(common.rooted_path(root, row["ligand_pdbqt"])):
            raise ValueError(f"Stage51 ligand retains closure pseudoatoms: {row['case_id']}")
    seeds = tuple((str(value["seed_id"]), int(value["base_seed"])) for value in inputs["seeds"])
    if seeds != EXPECTED_SEEDS:
        raise ValueError("Stage51 seed ledger differs")
    if int(expected["redocking_pair_count"]) != len(receptors) * len(seeds):
        raise ValueError("Stage51 redocking pair count differs")
    frozen_count = int(expected["frozen_receptor_count"])
    preparation_failures = [dict(value) for value in expected["preparation_failures"]]
    if frozen_count != len(receptors) + len(preparation_failures):
        raise ValueError("Stage51 frozen-pool accounting differs")
    observed_failures = [dict(value) for value in preparation["failed_cases"]]
    observed_failure_ids = [str(value["conformer_id"]) for value in observed_failures]
    expected_failure_ids = [str(value["conformer_id"]) for value in preparation_failures]
    if observed_failure_ids != expected_failure_ids:
        raise ValueError("Stage51 preparation-failure ledger differs")
    common.validate_protocol(dict(config["unidock"]), common_box)
    return receptors, cases, {
        "status": "audit_only_ok",
        "target_id": "PPARA",
        "receptor_count": len(receptors),
        "receptor_ids": receptor_ids,
        "case_count": len(cases),
        "seed_count": len(seeds),
        "expected_redocking_pair_count": len(receptors) * len(seeds),
        "frozen_receptor_count": frozen_count,
        "technical_preparation_failure_count": len(preparation_failures),
        "technical_preparation_failures": preparation_failures,
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
            raise ValueError(f"Stage51 receptor result count differs: {receptor_id}")
        rmsds = [float(row["top_ranked_rmsd_angstrom"]) for row in selected]
        successes = sum(common.truth(row["top_ranked_pose_success"]) for row in selected)
        median_rmsd = statistics.median(rmsds)
        output.append({
            "conformer_id": receptor_id,
            "seed_count": len(selected),
            "successful_seed_count": successes,
            "median_top_ranked_rmsd_angstrom": median_rmsd,
            "maximum_top_ranked_rmsd_angstrom": max(rmsds),
            "gate_pass": successes >= minimum_successes and median_rmsd <= threshold,
        })
    return output


def run(config_path: Path, root: Path, unidock: str | None, audit_only: bool, resume: bool) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = common.read_json(config_path)
    receptors, cases, input_audit = validate_inputs(root, config)
    config_hash = common.file_sha256(config_path)
    if audit_only:
        result = {
            "schema_version": "1.0", "experiment_id": config["experiment_id"],
            "config": {"path": common.relative_path(root, config_path), "sha256": config_hash},
            **input_audit, "operation": "input audit only; no GPU docking was started",
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    runtime = common.runtime_evidence(config)
    protocol = dict(config["unidock"])
    executable_info = common.executable_evidence(
        unidock or str(protocol["executable"]), str(protocol["required_package_version"])
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
            "ligand_id": case["case_id"], "label": "cognate_control",
            "selection_role": "technical_redocking", "pdbqt_path": case["ligand_pdbqt"],
            "pdbqt_sha256": case["ligand_pdbqt_sha256"],
        }
        for seed in dict(config["inputs"])["seeds"]:
            seed_id = str(seed["seed_id"])
            base_seed = int(seed["base_seed"])
            batch = common.batch_paths(run_root, seed_id, receptor["conformer_id"])
            batch["directory"].mkdir(parents=True, exist_ok=True)
            signature = common.protocol_signature(config_hash, seed_id, base_seed, receptor, [ligand], protocol)
            checkpoint = common.validate_checkpoint(root, batch, signature, {case["case_id"]}) if resume else None
            if checkpoint is not None:
                batch_rows, batch_summary = checkpoint
                if "warning_adjudication" not in batch_summary or not all("pose_integrity_status" in row for row in batch_rows):
                    checkpoint = None
            if checkpoint is None:
                print(f"running: {seed_id}/{receptor['conformer_id']}", flush=True)
                batch_rows, batch_summary = common.run_batch(
                    root, batch, str(executable_info["resolved_executable"]), receptor,
                    [ligand], protocol, seed_id, base_seed, signature,
                )
                for row in batch_rows:
                    row["target_id"] = "PPARA"
                batch_rows, pose_audit = common.audit_batch_poses(root, [ligand], batch_rows)
                warning = common.classify_warning_log(batch["log"], pose_audit)
                batch_summary["pose_integrity"] = pose_audit
                batch_summary["warning_adjudication"] = warning
                common.write_csv(batch["scores"], batch_rows)
                batch_summary["scores_sha256"] = common.file_sha256(batch["scores"])
                common.write_json(batch["summary"], batch_summary)
                executed += 1
            else:
                batch_rows, batch_summary = checkpoint
                resumed += 1
                print(f"resume ok: {seed_id}/{receptor['conformer_id']}", flush=True)
            output_pose = common.rooted_path(root, str(batch_rows[0]["output_pose_path"]))
            rmsd_dir = batch["directory"] / "rmsd"
            rmsd_summary_path = rmsd_dir / "summary.json"
            pose_table = rmsd_dir / "poses.csv"
            common.run_checked([
                sys.executable, str(common.rooted_path(root, str(dict(config["inputs"])["evaluate_rmsd_script"]["path"]))),
                "--case-id", f"{case['case_id']}__{seed_id}", "--reference-sdf",
                str(common.rooted_path(root, case["reference_sdf"])), "--docked-pdbqt", str(output_pose),
                "--pose-table-output", str(pose_table), "--summary-output", str(rmsd_summary_path),
                "--success-threshold", str(threshold),
            ], f"PPARA redocking RMSD evaluation for {case['case_id']} {seed_id}")
            rmsd = common.read_json(rmsd_summary_path)
            warning = dict(batch_summary["warning_adjudication"])
            pose_audit = dict(batch_summary["pose_integrity"])
            all_rows.append({
                "target_id": "PPARA", "case_id": case["case_id"],
                "conformer_id": receptor["conformer_id"], "seed_id": seed_id,
                "base_seed": base_seed, "reference_ligand": case["selected_ligand_resname"],
                "reference_sdf": case["reference_sdf"], "reference_sdf_sha256": case["reference_sdf_sha256"],
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
                "evaluation_summary_sha256": common.file_sha256(rmsd_summary_path), "status": "ok",
            })

    expected_pairs = int(dict(config["expected"])["redocking_pair_count"])
    if len(all_rows) != expected_pairs or len({(row["conformer_id"], row["seed_id"]) for row in all_rows}) != expected_pairs:
        raise ValueError("Stage51 complete redocking grid differs")
    gate = dict(config["redocking_gate"])
    gate_rows = summarize_gate(all_rows, input_audit["receptor_ids"], threshold, int(gate["minimum_successful_seeds_per_receptor"]))
    for failure in input_audit["technical_preparation_failures"]:
        gate_rows.append({
            "conformer_id": failure["conformer_id"],
            "seed_count": 0,
            "successful_seed_count": 0,
            "median_top_ranked_rmsd_angstrom": "",
            "maximum_top_ranked_rmsd_angstrom": "",
            "gate_pass": False,
            "failure_stage": "stage50_input_preparation",
            "failure_reason": failure["error"],
        })
    unresolved = sum(int(row["unresolved_warning_event_count"]) for row in all_rows)
    pose_failures = sum(int(row["pose_integrity_failure_count"]) for row in all_rows)
    passed = sum(common.truth(row["gate_pass"]) for row in gate_rows)
    minimum_passing = int(gate["minimum_passing_receptor_count"])
    status = "stage51_ppara_large_pool_cognate_redocking_gate_ok" if passed >= minimum_passing and unresolved == 0 and pose_failures == 0 else "stage51_ppara_large_pool_cognate_redocking_gate_failed"
    outputs = dict(config["outputs"])
    result_path = common.rooted_path(root, str(outputs["redocking_results_csv"]))
    gate_path = common.rooted_path(root, str(outputs["receptor_gate_results_csv"]))
    summary_path = common.rooted_path(root, str(outputs["summary_json"]))
    common.write_csv(result_path, all_rows)
    common.write_csv(gate_path, gate_rows)
    summary = {
        "schema_version": "1.0", "experiment_id": config["experiment_id"], "status": status,
        "config": {"path": common.relative_path(root, config_path), "sha256": config_hash},
        "runtime": runtime, "unidock_executable": executable_info, "frozen_protocol": protocol,
        "input_audit": input_audit, "redocking_pair_count": len(all_rows),
        "executed_batches_this_invocation": executed, "resumed_batches_this_invocation": resumed,
        "receptor_gate_results": gate_rows, "passed_receptor_count": passed,
        "frozen_receptor_count": input_audit["frozen_receptor_count"],
        "technical_preparation_failure_count": input_audit["technical_preparation_failure_count"],
        "failed_receptor_count": int(input_audit["frozen_receptor_count"]) - passed,
        "minimum_passing_receptor_count": minimum_passing,
        "technical_gate_pass": status.endswith("_ok"),
        "known_warning_event_count": sum(int(row["known_warning_event_count"]) for row in all_rows),
        "unresolved_warning_event_count": unresolved, "pose_integrity_failure_count": pose_failures,
        "data_boundary": {"ligand_labels_read": 0, "benchmark_docking_scores_read": 0, "fresh_validation_rows_read": 0, "test_rows_read": 0},
        "outputs": {
            "redocking_results_csv": {"path": common.relative_path(root, result_path), "sha256": common.file_sha256(result_path)},
            "receptor_gate_results_csv": {"path": common.relative_path(root, gate_path), "sha256": common.file_sha256(gate_path)},
        },
        "next_gate": "prepare the frozen 374-ligand PPARA development panel and dock it against every passing receptor without structural max-min compression" if status.endswith("_ok") else "stop the PPARA large-pool route and preserve all technical failures",
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
