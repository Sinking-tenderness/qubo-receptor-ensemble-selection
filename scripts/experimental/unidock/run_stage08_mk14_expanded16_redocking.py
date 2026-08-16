"""Prepare and Uni-Dock redock the eight new Stage 08 MAPK14 receptors."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import statistics
import sys
from pathlib import Path

try:
    from scripts.batch_prepare_ligand_pdbqt import find_meeko_script, parse_pdbqt
    from scripts.prepare_receptor import file_sha256
    from scripts.run_mk14_expanded_redocking_gate import (
        aligned_ligand_coordinates,
        alignment_transform_audit,
        box_audit,
        build_modelserver_url,
        coordinates_and_elements_from_sdf,
        download_cached,
        point_set_rmsd_by_element,
        run_checked,
    )
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
    from scripts.batch_prepare_ligand_pdbqt import find_meeko_script, parse_pdbqt
    from scripts.prepare_receptor import file_sha256
    from scripts.run_mk14_expanded_redocking_gate import (
        aligned_ligand_coordinates,
        alignment_transform_audit,
        box_audit,
        build_modelserver_url,
        coordinates_and_elements_from_sdf,
        download_cached,
        point_set_rmsd_by_element,
        run_checked,
    )
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


EXPECTED_PROTOCOL = {
    "required_package_version": "1.1.3",
    "profile_id": "enhanced",
    "scoring": "vina",
    "exhaustiveness": 1024,
    "max_step": 80,
    "refine_step": 5,
    "num_modes": 1,
    "energy_range": 3,
}


def checked_record(root: Path, record: dict[str, object]) -> Path:
    path = rooted_path(root, str(record["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(record["sha256"]).upper():
        raise ValueError(f"SHA-256 differs: {relative_path(root, path)}")
    return path


def verify_implementation(
    root: Path, config: dict[str, object], key: str, expected_path: Path
) -> None:
    descriptor = dict(config["implementation"])[key]
    path = rooted_path(root, str(descriptor["path"]))
    if path.resolve() != expected_path.resolve():
        raise ValueError(f"implementation path differs: {key}")
    if file_sha256(path) != str(descriptor["sha256"]).upper():
        raise ValueError(f"implementation SHA-256 differs: {key}")


def runtime_evidence(config: dict[str, object]) -> dict[str, str]:
    package_names = {
        "numpy_version": "numpy",
        "scipy_version": "scipy",
        "rdkit_version": "rdkit",
        "meeko_version": "meeko",
        "prody_version": "prody",
    }
    actual = {
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "python_version": platform.python_version(),
        **{
            key: importlib.metadata.version(package)
            for key, package in package_names.items()
        },
    }
    expected = {key: str(value) for key, value in dict(config["runtime"]).items()}
    if actual != expected:
        raise RuntimeError(f"runtime differs: {actual} != {expected}")
    return actual


def validate_protocol(protocol: dict[str, object]) -> None:
    for key, expected in EXPECTED_PROTOCOL.items():
        observed = protocol.get(key)
        if isinstance(expected, int):
            observed = int(observed)
        else:
            observed = str(observed)
        if observed != expected:
            raise ValueError(f"frozen Uni-Dock protocol differs: {key}")
    box = dict(protocol["box"])
    expected_box = {
        "center_x": -0.49,
        "center_y": 3.26,
        "center_z": 21.83,
        "size_x": 22,
        "size_y": 24,
        "size_z": 32,
    }
    if {key: float(box[key]) for key in expected_box} != {
        key: float(value) for key, value in expected_box.items()
    }:
        raise ValueError("frozen Stage 07c Uni-Dock box differs")


def validate_config(config: dict[str, object]) -> None:
    required = {
        "schema_version",
        "experiment_id",
        "purpose",
        "implementation",
        "runtime",
        "data_boundary",
        "inputs",
        "cases",
        "expected",
        "ligand_download",
        "reference_coordinate_gate",
        "common_box_gate",
        "unidock",
        "redocking_gate",
        "outputs",
        "decision_boundary",
    }
    if set(config) != required:
        raise ValueError("Stage 08 redocking config keys differ")
    boundary = dict(config["data_boundary"])
    if any(int(value) != 0 for value in boundary.values()):
        raise ValueError("Stage 08 redocking must retain a zero-use data boundary")
    validate_protocol(dict(config["unidock"]))
    seeds = [
        (str(row["seed_id"]), int(row["base_seed"]))
        for row in dict(config["inputs"])["seeds"]
    ]
    if seeds != [
        ("seed0", 20260801),
        ("seed1", 20260802),
        ("seed2", 20260803),
    ]:
        raise ValueError("Stage 08 paired seeds differ")


def validate_inputs(
    root: Path, config: dict[str, object]
) -> tuple[dict[str, Path], list[dict[str, str]], list[dict[str, str]], dict[str, object]]:
    validate_config(config)
    implementation_paths = {
        "runner": Path(__file__),
        "unidock_batch_helper": Path(__file__).with_name(
            "run_unidock_gpu_equivalence.py"
        ),
        "pose_audit_helper": Path(__file__).with_name(
            "run_stage07b_unidock_enhanced_confirmation.py"
        ),
        "warning_adjudication_helper": Path(__file__).with_name(
            "run_stage07c_unidock_warning_adjudication.py"
        ),
        "preparation_helper": root / "scripts/run_mk14_expanded_redocking_gate.py",
    }
    for key, path in implementation_paths.items():
        verify_implementation(root, config, key, path)

    inputs = dict(config["inputs"])
    paths = {
        key: checked_record(root, value)
        for key, value in inputs.items()
        if isinstance(value, dict)
    }
    required_paths = {
        "structural_summary",
        "structural_audit",
        "structural_manifest",
        "candidate_audit",
        "frozen_unidock_result",
        "existing8_receptor_manifest",
        "reference_pdb",
        "prepare_receptor_script",
        "transform_sdf_script",
        "evaluate_rmsd_script",
        "box_gate_amendment",
    }
    if set(paths) != required_paths:
        raise ValueError("Stage 08 redocking input paths differ")

    structural_summary = read_json(paths["structural_summary"])
    structural_audit = read_json(paths["structural_audit"])
    frozen_result = read_json(paths["frozen_unidock_result"])
    amendment = read_json(paths["box_gate_amendment"])
    if structural_summary.get("status") != "expanded16_structural_selection_ok":
        raise ValueError("Stage 08 structural selection did not pass")
    if structural_audit.get("status") != "independent_expanded16_structural_audit_ok":
        raise ValueError("Stage 08 independent structural audit did not pass")
    if frozen_result.get("status") != "unidock_profile_frozen_train_only":
        raise ValueError("Uni-Dock profile was not frozen")
    if frozen_result.get("selected_profile_id") != "enhanced":
        raise ValueError("frozen Uni-Dock profile is not enhanced")
    amendment_boundary = amendment.get("data_boundary")
    if not isinstance(amendment_boundary, dict) or any(
        int(value) != 0 for value in amendment_boundary.values()
    ):
        raise ValueError("box gate amendment crossed a data boundary")

    structural_rows = read_csv(paths["structural_manifest"])
    new_rows = sorted(
        (
            row
            for row in structural_rows
            if row["prefix_status"] == "new_expanded16_addition"
        ),
        key=lambda row: int(row["selection_rank"]),
    )
    expected_ids = [str(value) for value in dict(config["expected"])["new_receptor_ids"]]
    if [row["conformer_id"] for row in new_rows] != expected_ids:
        raise ValueError("Stage 08 new receptor order differs")

    cases = list(config["cases"])
    if [str(case["conformer_id"]) for case in cases] != expected_ids:
        raise ValueError("Stage 08 redocking case order differs")
    selected_by_id = {row["conformer_id"]: row for row in new_rows}
    candidate_audit_by_id = {
        row["conformer_id"]: row for row in read_csv(paths["candidate_audit"])
    }
    for case in cases:
        selected = selected_by_id[str(case["conformer_id"])]
        candidate_audit = candidate_audit_by_id.get(str(case["conformer_id"]))
        if candidate_audit is None or candidate_audit["status"] != "coordinate_eligible":
            raise ValueError(f"configured redocking case lacks a passing coordinate audit: {case['conformer_id']}")
        for key in (
            "pdb_id",
            "chain",
            "selected_ligand_resname",
            "selected_ligand_resseq",
            "selected_ligand_icode",
        ):
            if str(case.get(key, "")) != str(selected.get(key, "")):
                raise ValueError(f"configured redocking case differs: {case['conformer_id']} {key}")
        if str(case["selected_ligand_heavy_atom_count"]) != str(
            candidate_audit["selected_ligand_heavy_atom_count"]
        ):
            raise ValueError(
                f"configured redocking case differs: {case['conformer_id']} selected_ligand_heavy_atom_count"
            )
        for path_key, hash_key in (
            ("pdb_path", "pdb_sha256"),
            ("aligned_pdb_path", "aligned_pdb_sha256"),
        ):
            path = rooted_path(root, selected[path_key])
            if not path.is_file() or file_sha256(path) != selected[hash_key].upper():
                raise ValueError(f"selected coordinate hash differs: {case['conformer_id']}")

    existing_rows = read_csv(paths["existing8_receptor_manifest"])
    expected = dict(config["expected"])
    if len(existing_rows) != int(expected["existing_receptor_count"]):
        raise ValueError("existing receptor manifest count differs")
    if len({row["conformer_id"] for row in existing_rows}) != len(existing_rows):
        raise ValueError("existing receptor manifest contains duplicate IDs")
    for row in existing_rows:
        if row["status"] != "ok":
            raise ValueError("existing receptor manifest contains a failed row")
        path = rooted_path(root, row["receptor_pdbqt"])
        if not path.is_file() or file_sha256(path) != row["receptor_pdbqt_sha256"]:
            raise ValueError(f"existing receptor PDBQT hash differs: {row['conformer_id']}")

    expected_pairs = int(expected["new_receptor_count"]) * int(expected["seed_count"])
    if expected_pairs != int(expected["redocking_pair_count"]):
        raise ValueError("Stage 08 expected redocking pair count is inconsistent")
    audit = {
        "status": "audit_only_ok",
        "new_receptor_ids": expected_ids,
        "new_receptor_count": len(new_rows),
        "existing_receptor_count": len(existing_rows),
        "final_receptor_count": len(new_rows) + len(existing_rows),
        "seed_count": len(inputs["seeds"]),
        "expected_redocking_pair_count": expected_pairs,
        "frozen_profile_id": frozen_result["selected_profile_id"],
        "validation_rows": 0,
        "test_rows": 0,
    }
    return paths, new_rows, existing_rows, audit


def prepare_ligand(
    explicit_h_sdf: Path, output_path: Path
) -> tuple[dict[str, object], str]:
    command = [sys.executable, str(find_meeko_script()), "-i", str(explicit_h_sdf), "-o", str(output_path)]
    run_checked(command, f"ligand preparation for {output_path.stem}")
    pseudoatoms = macrocycle_closure_atom_types(output_path)
    variant = "meeko_flexible"
    if pseudoatoms:
        output_path.unlink()
        run_checked(
            command + ["--rigid_macrocycles"],
            f"rigid-macrocycle ligand preparation for {output_path.stem}",
        )
        variant = "meeko_rigid_macrocycles"
    remaining = macrocycle_closure_atom_types(output_path)
    if remaining:
        raise ValueError(f"Uni-Dock-incompatible macrocycle pseudoatoms remain: {remaining}")
    return parse_pdbqt(output_path), variant


def prepare_case(
    root: Path,
    config: dict[str, object],
    paths: dict[str, Path],
    selected: dict[str, str],
    case: dict[str, object],
    run_root: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    conformer_id = str(case["conformer_id"])
    case_id = f"{case['pdb_id']}_{case['selected_ligand_resname']}"
    case_root = run_root / "preparation" / case_id
    case_root.mkdir(parents=True, exist_ok=True)
    raw_pdb = rooted_path(root, selected["pdb_path"])
    aligned_pdb = rooted_path(root, selected["aligned_pdb_path"])

    alignment_path = case_root / "alignment_summary.json"
    alignment = alignment_transform_audit(
        paths["reference_pdb"], raw_pdb, aligned_pdb, str(case["chain"])
    )
    write_json(alignment_path, alignment)

    download = dict(config["ligand_download"])
    source_url = build_modelserver_url(
        str(download["url_template"]),
        str(case["pdb_id"]),
        str(case["chain"]),
        int(case["selected_ligand_resseq"]),
    )
    raw_sdf = case_root / f"{case_id}_raw.sdf"
    download_cached(
        source_url,
        raw_sdf,
        float(download["timeout_seconds"]),
        int(download["maximum_retries"]),
        float(download["retry_backoff_seconds"]),
    )

    common_sdf = case_root / f"{case_id}_common_frame.sdf"
    common_summary = case_root / "common_frame_transform.json"
    explicit_h_sdf = case_root / f"{case_id}_common_frame_explicitH.sdf"
    explicit_h_summary = case_root / "explicit_h_transform.json"
    for output_sdf, output_summary, add_hydrogens in (
        (common_sdf, common_summary, False),
        (explicit_h_sdf, explicit_h_summary, True),
    ):
        command = [
            sys.executable,
            str(paths["transform_sdf_script"]),
            "--input-sdf",
            str(raw_sdf),
            "--alignment-summary",
            str(alignment_path),
            "--output-sdf",
            str(output_sdf),
            "--summary-output",
            str(output_summary),
            "--overwrite",
        ]
        if add_hydrogens:
            command.append("--add-explicit-hydrogens")
        run_checked(command, f"common-frame transform for {case_id}")

    sdf_coords, sdf_elements, molecule = coordinates_and_elements_from_sdf(common_sdf)
    if molecule.GetNumHeavyAtoms() != int(case["selected_ligand_heavy_atom_count"]):
        raise ValueError(f"co-crystal heavy-atom count differs: {case_id}")
    pdb_coords, pdb_elements = aligned_ligand_coordinates(
        aligned_pdb,
        str(case["chain"]),
        str(case["selected_ligand_resname"]),
        int(case["selected_ligand_resseq"]),
        str(case.get("selected_ligand_icode", "")),
    )
    reference_rmsd, reference_maximum = point_set_rmsd_by_element(
        sdf_coords, sdf_elements, pdb_coords, pdb_elements
    )
    maximum_reference_rmsd = float(
        dict(config["reference_coordinate_gate"])[
            "maximum_element_matched_rmsd_angstrom"
        ]
    )
    if reference_rmsd > maximum_reference_rmsd:
        raise ValueError(f"transformed co-crystal coordinates differ: {case_id}")

    box = dict(config["unidock"])["box"]
    box_result = box_audit(
        sdf_coords,
        {axis: float(box[f"center_{axis}"]) for axis in ("x", "y", "z")},
        {axis: float(box[f"size_{axis}"]) for axis in ("x", "y", "z")},
    )

    receptor_root = case_root / "receptor"
    receptor_pdbqt = receptor_root / f"{conformer_id}_receptor.pdbqt"
    receptor_summary_path = receptor_root / "preparation_summary.json"
    run_checked(
        [
            sys.executable,
            str(paths["prepare_receptor_script"]),
            "--input-pdb",
            str(aligned_pdb),
            "--chain",
            str(case["chain"]),
            "--protein-only-output",
            str(receptor_root / f"{conformer_id}_protein_only.pdb"),
            "--prepared-pdb-output",
            str(receptor_root / f"{conformer_id}_prepared.pdb"),
            "--pdbqt-output",
            str(receptor_pdbqt),
            "--summary-output",
            str(receptor_summary_path),
            "--charge-model",
            "gasteiger",
            "--overwrite",
        ],
        f"receptor preparation for {case_id}",
    )
    receptor_summary = read_json(receptor_summary_path)
    if receptor_summary.get("status") != "ok":
        raise ValueError(f"receptor preparation failed: {case_id}")

    ligand_pdbqt = case_root / f"{case_id}.pdbqt"
    ligand_audit, preparation_variant = prepare_ligand(explicit_h_sdf, ligand_pdbqt)
    receptor_audit = receptor_summary["outputs"]["receptor_pdbqt"]["audit"]
    receptor_row = {
        "conformer_id": conformer_id,
        "source_pool": "stage08_new_structural_addition",
        "input_structure": relative_path(root, aligned_pdb),
        "input_structure_sha256": file_sha256(aligned_pdb),
        "chain": case["chain"],
        "residue_count": receptor_audit["residue_count"],
        "receptor_atom_count": receptor_audit["coordinate_record_count"],
        "hydrogen_like_atom_count": receptor_audit["hydrogen_like_atom_count"],
        "autodock_atom_types": ";".join(receptor_audit["autodock_atom_types"]),
        "charge_min": receptor_audit["charge_min"],
        "charge_max": receptor_audit["charge_max"],
        "receptor_pdbqt": relative_path(root, receptor_pdbqt),
        "receptor_pdbqt_sha256": file_sha256(receptor_pdbqt),
        "status": "ok",
    }
    prepared = {
        "case_id": case_id,
        "conformer_id": conformer_id,
        "reference_ligand": case["selected_ligand_resname"],
        "source_url": source_url,
        "raw_sdf": relative_path(root, raw_sdf),
        "raw_sdf_sha256": file_sha256(raw_sdf),
        "reference_sdf": relative_path(root, common_sdf),
        "reference_sdf_sha256": file_sha256(common_sdf),
        "explicit_h_sdf": relative_path(root, explicit_h_sdf),
        "explicit_h_sdf_sha256": file_sha256(explicit_h_sdf),
        "ligand_pdbqt": relative_path(root, ligand_pdbqt),
        "ligand_pdbqt_sha256": file_sha256(ligand_pdbqt),
        "ligand_pdbqt_audit": ligand_audit,
        "ligand_preparation_variant": preparation_variant,
        "receptor_pdbqt": relative_path(root, receptor_pdbqt),
        "receptor_pdbqt_sha256": file_sha256(receptor_pdbqt),
        "alignment_summary": relative_path(root, alignment_path),
        "alignment_summary_sha256": file_sha256(alignment_path),
        "reference_coordinate_rmsd_angstrom": reference_rmsd,
        "reference_coordinate_maximum_distance_angstrom": reference_maximum,
        "box_audit": box_result,
    }
    return receptor_row, prepared


def merge_receptor_manifests(
    existing_rows: list[dict[str, str]], new_rows: list[dict[str, object]]
) -> list[dict[str, object]]:
    fields = list(existing_rows[0])
    for row in new_rows:
        if set(row) != set(fields):
            raise ValueError("new receptor manifest schema differs from existing schema")
    combined: list[dict[str, object]] = [dict(row) for row in existing_rows]
    combined.extend({field: row[field] for field in fields} for row in new_rows)
    if len({str(row["conformer_id"]) for row in combined}) != len(combined):
        raise ValueError("combined receptor manifest contains duplicate IDs")
    return combined


def summarize_receptor_redocking_gate(
    rows: list[dict[str, object]],
    receptor_ids: list[str],
    threshold: float,
    minimum_successes: int,
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for receptor_id in receptor_ids:
        receptor_rows = [
            row for row in rows if str(row["conformer_id"]) == receptor_id
        ]
        if not receptor_rows:
            raise ValueError(f"redocking results are missing receptor: {receptor_id}")
        rmsds = [float(row["top_ranked_rmsd_angstrom"]) for row in receptor_rows]
        successes = sum(
            str(row["top_ranked_pose_success"]).lower() == "true"
            for row in receptor_rows
        )
        median_rmsd = statistics.median(rmsds)
        passed = successes >= minimum_successes and median_rmsd <= threshold
        summaries.append(
            {
                "conformer_id": receptor_id,
                "seed_count": len(receptor_rows),
                "successful_seed_count": successes,
                "median_top_ranked_rmsd_angstrom": median_rmsd,
                "maximum_top_ranked_rmsd_angstrom": max(rmsds),
                "gate_pass": passed,
            }
        )
    return summaries


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
    paths, selected_rows, existing_rows, input_audit = validate_inputs(root, config)
    if audit_only:
        result = {
            "schema_version": "1.0",
            "experiment_id": config["experiment_id"],
            "config": {
                "path": relative_path(root, config_path),
                "sha256": file_sha256(config_path),
            },
            **input_audit,
            "operation": "input and frozen-protocol audit only; no preparation, download, or GPU docking was started",
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return result

    runtime = runtime_evidence(config)
    protocol = dict(config["unidock"])
    executable = unidock or str(protocol["executable"])
    executable_info = executable_evidence(
        executable, str(protocol["required_package_version"])
    )
    outputs = dict(config["outputs"])
    run_root = rooted_path(root, str(outputs["run_directory"]))
    run_root.mkdir(parents=True, exist_ok=True)
    config_sha256 = file_sha256(config_path)
    selected_by_id = {row["conformer_id"]: row for row in selected_rows}

    new_receptor_rows: list[dict[str, object]] = []
    prepared_cases: list[dict[str, object]] = []
    for case in config["cases"]:
        conformer_id = str(case["conformer_id"])
        receptor_row, prepared = prepare_case(
            root,
            config,
            paths,
            selected_by_id[conformer_id],
            dict(case),
            run_root,
        )
        new_receptor_rows.append(receptor_row)
        prepared_cases.append(prepared)

    new_manifest_path = rooted_path(root, str(outputs["new_receptor_manifest_csv"]))
    write_csv(new_manifest_path, new_receptor_rows)
    minimum_margin = float(dict(config["common_box_gate"])["minimum_crystal_pose_margin_angstrom"])
    box_failures = [
        row
        for row in prepared_cases
        if float(row["box_audit"]["minimum_margin_angstrom"]) < minimum_margin
    ]
    if box_failures:
        raise ValueError(
            "Stage 08 common-box gate failed: "
            + ", ".join(str(row["case_id"]) for row in box_failures)
        )

    receptor_by_id = {str(row["conformer_id"]): row for row in new_receptor_rows}
    seeds = list(dict(config["inputs"])["seeds"])
    threshold = float(dict(config["redocking_gate"])["maximum_rmsd_angstrom"])
    all_rows: list[dict[str, object]] = []
    executed_batches = 0
    resumed_batches = 0
    for prepared in prepared_cases:
        receptor_id = str(prepared["conformer_id"])
        receptor = receptor_by_id[receptor_id]
        ligand = {
            "ligand_id": str(prepared["case_id"]),
            "label": "cognate_control",
            "selection_role": "technical_redocking",
            "pdbqt_path": str(prepared["ligand_pdbqt"]),
            "pdbqt_sha256": str(prepared["ligand_pdbqt_sha256"]),
        }
        for seed in seeds:
            seed_id = str(seed["seed_id"])
            base_seed = int(seed["base_seed"])
            batch = batch_paths(run_root / "redocking", seed_id, receptor_id)
            batch["directory"].mkdir(parents=True, exist_ok=True)
            signature = protocol_signature(
                config_sha256,
                seed_id,
                base_seed,
                receptor,
                [ligand],
                protocol,
            )
            checkpoint = (
                validate_checkpoint(root, batch, signature, {ligand["ligand_id"]})
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
                print(f"running: {seed_id}/{receptor_id}", flush=True)
                rows, batch_summary = run_batch(
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
                rows, pose_audit = audit_batch_poses(root, [ligand], rows)
                warning = classify_warning_log(batch["log"], pose_audit)
                batch_summary["pose_integrity"] = pose_audit
                batch_summary["warning_adjudication"] = warning
                write_csv(batch["scores"], rows)
                batch_summary["scores_sha256"] = file_sha256(batch["scores"])
                write_json(batch["summary"], batch_summary)
                executed_batches += 1
            else:
                rows, batch_summary = checkpoint
                resumed_batches += 1
                print(f"resume ok: {seed_id}/{receptor_id}", flush=True)

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
                f"redocking RMSD evaluation for {prepared['case_id']} {seed_id}",
            )
            evaluation = read_json(evaluation_path)
            warning = dict(batch_summary["warning_adjudication"])
            pose_audit = dict(batch_summary["pose_integrity"])
            all_rows.append(
                {
                    "case_id": prepared["case_id"],
                    "conformer_id": receptor_id,
                    "seed_id": seed_id,
                    "base_seed": base_seed,
                    "reference_ligand": prepared["reference_ligand"],
                    "reference_sdf": prepared["reference_sdf"],
                    "reference_sdf_sha256": prepared["reference_sdf_sha256"],
                    "receptor_pdbqt": receptor["receptor_pdbqt"],
                    "receptor_pdbqt_sha256": receptor["receptor_pdbqt_sha256"],
                    "docked_pdbqt": relative_path(root, output_pose),
                    "docked_pdbqt_sha256": file_sha256(output_pose),
                    "top_ranked_affinity_kcal_per_mol": evaluation[
                        "top_ranked_affinity_kcal_per_mol"
                    ],
                    "top_ranked_rmsd_angstrom": evaluation[
                        "top_ranked_rmsd_angstrom"
                    ],
                    "top_ranked_pose_success": evaluation[
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
                    "evaluation_summary": relative_path(root, evaluation_path),
                    "evaluation_summary_sha256": file_sha256(evaluation_path),
                    "status": "ok",
                }
            )

    expected_pairs = int(dict(config["expected"])["redocking_pair_count"])
    if len(all_rows) != expected_pairs:
        raise ValueError("Stage 08 redocking result count differs")
    result_keys = {
        (str(row["conformer_id"]), str(row["seed_id"])) for row in all_rows
    }
    if len(result_keys) != expected_pairs:
        raise ValueError("Stage 08 redocking results contain duplicate pairs")

    gate = dict(config["redocking_gate"])
    minimum_successes = int(gate["minimum_successful_seeds_per_receptor"])
    receptor_gate_rows = summarize_receptor_redocking_gate(
        all_rows,
        [str(value) for value in input_audit["new_receptor_ids"]],
        threshold,
        minimum_successes,
    )

    unresolved_warnings = sum(
        int(row["unresolved_warning_event_count"]) for row in all_rows
    )
    pose_failures = sum(int(row["pose_integrity_failure_count"]) for row in all_rows)
    all_receptors_pass = all(bool(row["gate_pass"]) for row in receptor_gate_rows)
    status = (
        "expanded16_unidock_redocking_gate_ok"
        if all_receptors_pass and unresolved_warnings == 0 and pose_failures == 0
        else "expanded16_unidock_redocking_gate_failed"
    )

    results_path = rooted_path(root, str(outputs["redocking_results_csv"]))
    write_csv(results_path, all_rows)
    combined_path = rooted_path(root, str(outputs["combined_receptor_manifest_csv"]))
    combined_rows = merge_receptor_manifests(existing_rows, new_receptor_rows)
    if status == "expanded16_unidock_redocking_gate_ok":
        write_csv(combined_path, combined_rows)

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
        "case_preparation": prepared_cases,
        "box_gate": {
            "minimum_required_margin_angstrom": minimum_margin,
            "minimum_observed_margin_angstrom": min(
                float(row["box_audit"]["minimum_margin_angstrom"])
                for row in prepared_cases
            ),
            "failure_count": len(box_failures),
        },
        "redocking_pair_count": len(all_rows),
        "executed_batches_this_invocation": executed_batches,
        "resumed_batches_this_invocation": resumed_batches,
        "receptor_gate_results": receptor_gate_rows,
        "all_new_receptors_pass": all_receptors_pass,
        "known_warning_event_count": sum(
            int(row["known_warning_event_count"]) for row in all_rows
        ),
        "unresolved_warning_event_count": unresolved_warnings,
        "pose_integrity_failure_count": pose_failures,
        "data_boundary": {
            "ligand_labels_read": 0,
            "benchmark_docking_scores_read": 0,
            "previous_validation_rows_read": 0,
            "test_rows_read": 0,
        },
        "outputs": {
            "new_receptor_manifest_csv": {
                "path": relative_path(root, new_manifest_path),
                "sha256": file_sha256(new_manifest_path),
            },
            "redocking_results_csv": {
                "path": relative_path(root, results_path),
                "sha256": file_sha256(results_path),
            },
            "combined_receptor_manifest_csv": (
                {
                    "path": relative_path(root, combined_path),
                    "sha256": file_sha256(combined_path),
                }
                if combined_path.is_file()
                else None
            ),
        },
        "next_gate": (
            "run the independent redocking audit; only then may a Train-696 x 16 x 3 Uni-Dock production bundle be created"
            if status == "expanded16_unidock_redocking_gate_ok"
            else "stop before production docking and review failed receptor cases"
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
        "expanded16_unidock_redocking_gate_ok",
    } else 2


if __name__ == "__main__":
    raise SystemExit(main())
