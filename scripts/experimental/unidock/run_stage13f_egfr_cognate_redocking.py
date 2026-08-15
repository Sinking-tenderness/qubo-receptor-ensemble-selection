"""Run and adjudicate the frozen EGFR 16-receptor cognate-redocking gate."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path

try:
    from scripts.prepare_receptor import file_sha256
    from scripts.experimental.unidock.run_stage07b_unidock_enhanced_confirmation import (
        audit_batch_poses,
    )
    from scripts.experimental.unidock.run_stage07c_unidock_warning_adjudication import (
        classify_warning_log,
    )
    from scripts.experimental.unidock.run_unidock_gpu_equivalence import (
        batch_paths,
        executable_evidence,
        macrocycle_closure_atom_types,
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
    from scripts.experimental.unidock.run_stage07b_unidock_enhanced_confirmation import (
        audit_batch_poses,
    )
    from scripts.experimental.unidock.run_stage07c_unidock_warning_adjudication import (
        classify_warning_log,
    )
    from scripts.experimental.unidock.run_unidock_gpu_equivalence import (
        batch_paths,
        executable_evidence,
        macrocycle_closure_atom_types,
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


EXPECTED_SEEDS = (
    ("seed0", 20260801),
    ("seed1", 20260802),
    ("seed2", 20260803),
)
EXPECTED_PROTOCOL = {
    "required_package_version": "1.1.3",
    "profile_id": "enhanced",
    "scoring": "vina",
    "exhaustiveness": 1024,
    "max_step": 80,
    "refine_step": 5,
    "num_modes": 1,
    "energy_range": 3,
    "verbosity": 1,
    "cuda_visible_devices": "0",
    "maximum_absolute_score_kcal_per_mol": 100.0,
}


def run_checked(command: list[str], operation: str) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        combined = "\n".join(
            value.strip()
            for value in (completed.stdout, completed.stderr)
            if value.strip()
        )
        raise RuntimeError(
            f"{operation} failed with return code {completed.returncode}: "
            f"{combined[-2000:]}"
        )
    return completed


def verified(root: Path, descriptor: dict[str, object]) -> Path:
    path = rooted_path(root, str(descriptor["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"SHA-256 differs: {relative_path(root, path)}")
    return path


def verify_implementation(
    root: Path, config: dict[str, object], key: str, expected: Path
) -> None:
    descriptor = dict(dict(config["implementation"])[key])
    path = rooted_path(root, str(descriptor["path"]))
    if path.resolve() != expected.resolve():
        raise ValueError(f"Stage 13f implementation path differs: {key}")
    if file_sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"Stage 13f implementation hash differs: {key}")


def runtime_evidence(config: dict[str, object]) -> dict[str, str]:
    packages = {
        "numpy_version": "numpy",
        "scipy_version": "scipy",
        "rdkit_version": "rdkit",
        "meeko_version": "meeko",
    }
    actual = {
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "python_version": platform.python_version(),
        **{
            key: importlib.metadata.version(package)
            for key, package in packages.items()
        },
    }
    expected = {key: str(value) for key, value in dict(config["runtime"]).items()}
    if actual != expected:
        raise RuntimeError(f"Stage 13f runtime differs: {actual} != {expected}")
    return actual


def validate_protocol(
    protocol: dict[str, object], common_box: dict[str, object]
) -> None:
    for key, expected in EXPECTED_PROTOCOL.items():
        observed = protocol.get(key)
        if isinstance(expected, int):
            observed = int(observed)
        elif isinstance(expected, float):
            observed = float(observed)
        else:
            observed = str(observed)
        if observed != expected:
            raise ValueError(f"Stage 13f frozen protocol differs: {key}")
    box = dict(protocol["box"])
    center = dict(common_box["center"])
    size = dict(common_box["size"])
    expected_box = {
        **{f"center_{axis}": float(center[axis]) for axis in ("x", "y", "z")},
        **{f"size_{axis}": float(size[axis]) for axis in ("x", "y", "z")},
    }
    if {key: float(box[key]) for key in expected_box} != expected_box:
        raise ValueError("Stage 13f Uni-Dock box differs from Stage 13e")


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

    boundary = dict(config["data_boundary"])
    if any(int(value) != 0 for value in boundary.values()):
        raise ValueError("Stage 13f crossed a frozen data boundary")
    inputs = dict(config["inputs"])
    paths = {
        key: verified(root, dict(value))
        for key, value in inputs.items()
        if isinstance(value, dict)
    }
    preparation = read_json(paths["preparation_summary"])
    common_box = read_json(paths["common_box"])
    if preparation.get("status") != "stage13e_egfr_redocking_inputs_ok":
        raise ValueError("Stage 13e input preparation did not pass")
    if any(int(value) != 0 for value in dict(preparation["data_boundary"]).values()):
        raise ValueError("Stage 13e preparation crossed a data boundary")
    for key in ("receptor_manifest_csv", "redocking_case_manifest_csv", "common_box_json"):
        source = dict(preparation["outputs"])[key]
        if file_sha256(rooted_path(root, str(source["path"]))) != str(
            source["sha256"]
        ).upper():
            raise ValueError(f"Stage 13e output identity differs: {key}")

    receptors = read_csv(paths["receptor_manifest"])
    cases = read_csv(paths["case_manifest"])
    expected = dict(config["expected"])
    receptor_ids = [row["conformer_id"] for row in receptors]
    if receptor_ids != [str(value) for value in expected["receptor_ids"]]:
        raise ValueError("Stage 13f receptor order differs")
    if len(receptors) != int(expected["receptor_count"]):
        raise ValueError("Stage 13f receptor count differs")
    if len(cases) != int(expected["case_count"]):
        raise ValueError("Stage 13f case count differs")
    if [row["conformer_id"] for row in cases] != receptor_ids:
        raise ValueError("Stage 13f receptor and cognate-case order differs")
    if len({row["case_id"] for row in cases}) != len(cases):
        raise ValueError("Stage 13f case IDs are not unique")
    if any(row["status"] != "ok" for row in receptors + cases):
        raise ValueError("Stage 13f input manifest contains a failed row")

    for row in receptors:
        path = rooted_path(root, row["receptor_pdbqt"])
        if not path.is_file() or file_sha256(path) != row[
            "receptor_pdbqt_sha256"
        ].upper():
            raise ValueError(f"Stage 13f receptor identity differs: {row['conformer_id']}")
    for row in cases:
        for path_key, hash_key in (
            ("ligand_pdbqt", "ligand_pdbqt_sha256"),
            ("reference_sdf", "reference_sdf_sha256"),
        ):
            path = rooted_path(root, row[path_key])
            if not path.is_file() or file_sha256(path) != row[hash_key].upper():
                raise ValueError(f"Stage 13f case identity differs: {row['case_id']}")
        pseudoatoms = macrocycle_closure_atom_types(
            rooted_path(root, row["ligand_pdbqt"])
        )
        if pseudoatoms:
            raise ValueError(
                f"Stage 13f ligand retains closure pseudoatoms: {row['case_id']}"
            )

    seeds = tuple(
        (str(value["seed_id"]), int(value["base_seed"]))
        for value in inputs["seeds"]
    )
    if seeds != EXPECTED_SEEDS:
        raise ValueError("Stage 13f seed ledger differs")
    if int(expected["redocking_pair_count"]) != len(receptors) * len(seeds):
        raise ValueError("Stage 13f redocking pair count differs")
    validate_protocol(dict(config["unidock"]), common_box)
    return receptors, cases, {
        "status": "audit_only_ok",
        "target_id": "EGFR",
        "receptor_count": len(receptors),
        "receptor_ids": receptor_ids,
        "case_count": len(cases),
        "seed_count": len(seeds),
        "expected_redocking_pair_count": len(receptors) * len(seeds),
        "common_box": common_box,
        "citation_intent_diagnostic": preparation["citation_intent_diagnostic"],
        "ligand_labels_read": 0,
        "benchmark_docking_scores_read": 0,
        "fresh_validation_rows_read": 0,
        "test_rows_read": 0,
    }


def truth(value: object) -> bool:
    return value is True or str(value).lower() == "true"


def summarize_gate(
    rows: list[dict[str, object]],
    receptor_ids: list[str],
    threshold: float,
    minimum_successes: int,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for receptor_id in receptor_ids:
        selected = [row for row in rows if row["conformer_id"] == receptor_id]
        if len(selected) != len(EXPECTED_SEEDS):
            raise ValueError(f"Stage 13f receptor result count differs: {receptor_id}")
        rmsds = [float(row["top_ranked_rmsd_angstrom"]) for row in selected]
        successes = sum(truth(row["top_ranked_pose_success"]) for row in selected)
        median_rmsd = statistics.median(rmsds)
        output.append(
            {
                "conformer_id": receptor_id,
                "seed_count": len(selected),
                "successful_seed_count": successes,
                "median_top_ranked_rmsd_angstrom": median_rmsd,
                "maximum_top_ranked_rmsd_angstrom": max(rmsds),
                "gate_pass": successes >= minimum_successes
                and median_rmsd <= threshold,
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
    config = read_json(config_path)
    receptors, cases, input_audit = validate_inputs(root, config)
    config_hash = file_sha256(config_path)
    if audit_only:
        result = {
            "schema_version": "1.0",
            "experiment_id": config["experiment_id"],
            "config": {
                "path": relative_path(root, config_path),
                "sha256": config_hash,
            },
            **input_audit,
            "operation": "input audit only; no GPU docking was started",
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    runtime = runtime_evidence(config)
    protocol = dict(config["unidock"])
    executable = unidock or str(protocol["executable"])
    executable_info = executable_evidence(
        executable, str(protocol["required_package_version"])
    )
    run_root = rooted_path(root, str(dict(config["outputs"])["run_directory"]))
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
            batch = batch_paths(run_root, seed_id, receptor["conformer_id"])
            batch["directory"].mkdir(parents=True, exist_ok=True)
            signature = protocol_signature(
                config_hash,
                seed_id,
                base_seed,
                receptor,
                [ligand],
                protocol,
            )
            checkpoint = (
                validate_checkpoint(root, batch, signature, {case["case_id"]})
                if resume
                else None
            )
            if checkpoint is not None:
                batch_rows, batch_summary = checkpoint
                if "warning_adjudication" not in batch_summary or not all(
                    "pose_integrity_status" in row for row in batch_rows
                ):
                    checkpoint = None
            if checkpoint is None:
                print(f"running: {seed_id}/{receptor['conformer_id']}", flush=True)
                batch_rows, batch_summary = run_batch(
                    root,
                    batch,
                    str(executable_info["resolved_executable"]),
                    receptor,
                    [ligand],
                    protocol,
                    seed_id,
                    base_seed,
                    signature,
                )
                for row in batch_rows:
                    row["target_id"] = "EGFR"
                batch_rows, pose_audit = audit_batch_poses(
                    root, [ligand], batch_rows
                )
                warning = classify_warning_log(batch["log"], pose_audit)
                batch_summary["pose_integrity"] = pose_audit
                batch_summary["warning_adjudication"] = warning
                write_csv(batch["scores"], batch_rows)
                batch_summary["scores_sha256"] = file_sha256(batch["scores"])
                write_json(batch["summary"], batch_summary)
                executed += 1
            else:
                batch_rows, batch_summary = checkpoint
                resumed += 1
                print(f"resume ok: {seed_id}/{receptor['conformer_id']}", flush=True)

            output_pose = rooted_path(root, str(batch_rows[0]["output_pose_path"]))
            rmsd_dir = batch["directory"] / "rmsd"
            rmsd_summary_path = rmsd_dir / "summary.json"
            pose_table = rmsd_dir / "poses.csv"
            run_checked(
                [
                    sys.executable,
                    str(rooted_path(root, str(dict(config["inputs"])["evaluate_rmsd_script"]["path"]))),
                    "--case-id",
                    f"{case['case_id']}__{seed_id}",
                    "--reference-sdf",
                    str(rooted_path(root, case["reference_sdf"])),
                    "--docked-pdbqt",
                    str(output_pose),
                    "--pose-table-output",
                    str(pose_table),
                    "--summary-output",
                    str(rmsd_summary_path),
                    "--success-threshold",
                    str(threshold),
                ],
                f"EGFR redocking RMSD evaluation for {case['case_id']} {seed_id}",
            )
            rmsd = read_json(rmsd_summary_path)
            warning = dict(batch_summary["warning_adjudication"])
            pose_audit = dict(batch_summary["pose_integrity"])
            all_rows.append(
                {
                    "target_id": "EGFR",
                    "case_id": case["case_id"],
                    "conformer_id": receptor["conformer_id"],
                    "seed_id": seed_id,
                    "base_seed": base_seed,
                    "reference_ligand": case["selected_ligand_resname"],
                    "reference_sdf": case["reference_sdf"],
                    "reference_sdf_sha256": case["reference_sdf_sha256"],
                    "docked_pdbqt": relative_path(root, output_pose),
                    "docked_pdbqt_sha256": file_sha256(output_pose),
                    "top_ranked_affinity_kcal_per_mol": rmsd[
                        "top_ranked_affinity_kcal_per_mol"
                    ],
                    "top_ranked_rmsd_angstrom": rmsd[
                        "top_ranked_rmsd_angstrom"
                    ],
                    "top_ranked_pose_success": rmsd[
                        "top_ranked_pose_success"
                    ],
                    "pose_integrity_failure_count": pose_audit["failure_count"],
                    "known_warning_event_count": warning[
                        "known_warning_event_count"
                    ],
                    "unresolved_warning_event_count": warning[
                        "unresolved_warning_event_count"
                    ],
                    "batch_summary": relative_path(root, batch["summary"]),
                    "batch_summary_sha256": file_sha256(batch["summary"]),
                    "evaluation_summary": relative_path(root, rmsd_summary_path),
                    "evaluation_summary_sha256": file_sha256(rmsd_summary_path),
                    "status": "ok",
                }
            )

    expected_pairs = int(dict(config["expected"])["redocking_pair_count"])
    if len(all_rows) != expected_pairs or len(
        {(row["conformer_id"], row["seed_id"]) for row in all_rows}
    ) != expected_pairs:
        raise ValueError("Stage 13f complete redocking grid differs")
    gate = dict(config["redocking_gate"])
    gate_rows = summarize_gate(
        all_rows,
        input_audit["receptor_ids"],
        threshold,
        int(gate["minimum_successful_seeds_per_receptor"]),
    )
    unresolved = sum(int(row["unresolved_warning_event_count"]) for row in all_rows)
    pose_failures = sum(int(row["pose_integrity_failure_count"]) for row in all_rows)
    all_pass = all(truth(row["gate_pass"]) for row in gate_rows)
    status = (
        "stage13f_egfr_cognate_redocking_gate_ok"
        if all_pass and unresolved == 0 and pose_failures == 0
        else "stage13f_egfr_cognate_redocking_gate_failed"
    )
    outputs = dict(config["outputs"])
    result_path = rooted_path(root, str(outputs["redocking_results_csv"]))
    gate_path = rooted_path(root, str(outputs["receptor_gate_results_csv"]))
    summary_path = rooted_path(root, str(outputs["summary_json"]))
    write_csv(result_path, all_rows)
    write_csv(gate_path, gate_rows)
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": status,
        "config": {
            "path": relative_path(root, config_path),
            "sha256": config_hash,
        },
        "runtime": runtime,
        "unidock_executable": executable_info,
        "frozen_protocol": protocol,
        "input_audit": input_audit,
        "redocking_pair_count": len(all_rows),
        "executed_batches_this_invocation": executed,
        "resumed_batches_this_invocation": resumed,
        "receptor_gate_results": gate_rows,
        "passed_receptor_count": sum(truth(row["gate_pass"]) for row in gate_rows),
        "failed_receptor_count": sum(not truth(row["gate_pass"]) for row in gate_rows),
        "all_receptors_pass": all_pass,
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
                "path": relative_path(root, result_path),
                "sha256": file_sha256(result_path),
            },
            "receptor_gate_results_csv": {
                "path": relative_path(root, gate_path),
                "sha256": file_sha256(gate_path),
            },
        },
        "next_gate": (
            "prepare the untouched EGFR Train-696 ligand panel and run 696 x 16 x 3 Uni-Dock production"
            if status == "stage13f_egfr_cognate_redocking_gate_ok"
            else "stop benchmark docking and adjudicate failed receptors without using activity labels"
        ),
        "decision_boundary": config["decision_boundary"],
    }
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
    run(args.config, args.root, args.unidock, args.audit_only, args.resume)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
