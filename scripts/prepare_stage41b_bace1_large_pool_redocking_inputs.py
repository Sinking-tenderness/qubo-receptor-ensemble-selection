"""Prepare the frozen 49-receptor BACE1 large-pool redocking inputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.freeze_stage41a_bace1_large_pool import (
    file_sha256,
    read_csv,
    read_json,
    rooted,
    write_csv,
    write_json,
)


def verified(path: Path, expected_sha256: str) -> Path:
    if not path.is_file() or file_sha256(path) != expected_sha256.upper():
        raise ValueError(f"input identity differs: {path}")
    return path


def load_preparation_dependencies() -> None:
    global alignment_audit_fa10
    global build_modelserver_url
    global coordinates_and_elements_from_sdf_safe
    global derive_common_box
    global download_cached
    global ligand_residue_map
    global point_set_rmsd_by_element
    global prepare_ligand_safe
    global run_checked
    global transform_sdf_safe

    from scripts.prepare_stage18d_pparg_redocking_inputs import (
        alignment_audit_fa10,
        build_modelserver_url,
        coordinates_and_elements_from_sdf_safe,
        derive_common_box,
        download_cached,
        ligand_residue_map,
        point_set_rmsd_by_element,
        prepare_ligand_safe,
        run_checked,
        transform_sdf_safe,
    )


def verify_descriptor(root: Path, descriptor: dict[str, object]) -> Path:
    return verified(rooted(root, str(descriptor["path"])), str(descriptor["sha256"]))


def validate_inputs(
    config_path: Path, root: Path
) -> tuple[dict[str, object], dict[str, Path], list[dict[str, str]], dict[str, object]]:
    config = read_json(config_path)
    implementation = dict(config["implementation"])
    if file_sha256(Path(__file__)) != str(implementation["sha256"]).upper():
        raise ValueError("Stage41b implementation SHA-256 differs")
    for dependency in config.get("dependencies", []):
        verify_descriptor(root, dict(dependency))
    inputs = {
        key: verify_descriptor(root, dict(value))
        for key, value in dict(config["inputs"]).items()
    }
    freeze = read_json(inputs["stage41a_result"])
    audit = read_json(inputs["stage41a_audit"])
    if freeze.get("status") != "stage41a_bace1_large_pool_frozen":
        raise ValueError("Stage41a large-pool freeze did not pass")
    if audit.get("status") != "stage41a_bace1_large_pool_freeze_audit_ok":
        raise ValueError("Stage41a independent audit did not pass")
    if not all(dict(audit["checks"]).values()):
        raise ValueError("Stage41a audit contains a failed check")
    rows = read_csv(inputs["large_pool_manifest"])
    expected = int(dict(config["expected"])["receptor_count"])
    if len(rows) != expected or len({row["conformer_id"] for row in rows}) != expected:
        raise ValueError("Stage41b frozen receptor count or identity differs")
    if rows[0]["conformer_id"] != "BACE1_3L5D_reference":
        raise ValueError("Stage41b reference receptor is not first")
    if len({(row["pdb_id"], row["selected_ligand_resname"], row["selected_ligand_resseq"]) for row in rows}) != expected:
        raise ValueError("Stage41b cognate cases are not unique")
    for row in rows:
        verified(rooted(root, row["mmcif_path"]), row["mmcif_sha256"])
        verified(
            rooted(root, row["aligned_protein_pdb_path"]),
            row["aligned_protein_pdb_sha256"],
        )
    overrides = dict(config.get("receptor_preparation_overrides", {}))
    unknown_overrides = set(overrides).difference(row["conformer_id"] for row in rows)
    if unknown_overrides:
        raise ValueError(f"Stage41b override references unknown receptors: {sorted(unknown_overrides)}")
    for conformer_id, value in overrides.items():
        override = dict(value)
        if set(override) != {
            "set_template",
            "blunt_ends",
            "evidence",
            "interpretation",
        }:
            raise ValueError(f"Stage41b override schema differs: {conformer_id}")
        if not str(override["set_template"]).strip() or override["blunt_ends"] is not None:
            raise ValueError(f"Stage41b override is not the frozen CYX-only correction: {conformer_id}")
    boundary = {
        "ligand_labels_read": 0,
        "benchmark_docking_scores_read": 0,
        "fresh_validation_rows_read": 0,
        "locked_test_rows_read": 0,
        "new_docking_jobs": 0,
        "quantum_hardware_jobs": 0,
    }
    return config, inputs, rows, {
        "status": "audit_only_ok",
        "target_id": "BACE1",
        "receptor_count": len(rows),
        "cognate_ligand_count": len(rows),
        "receptor_ids": [row["conformer_id"] for row in rows],
        "data_boundary": boundary,
    }


def checkpoint(
    root: Path, path: Path
) -> tuple[dict[str, object], dict[str, object], np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        record = read_json(path)
        if record.get("status") != "stage41b_case_preparation_ok":
            return None
        receptor = dict(record["receptor_row"])
        case = dict(record["case_row"])
        for path_key, hash_key in (
            ("receptor_pdbqt", "receptor_pdbqt_sha256"),
            ("receptor_preparation_summary", "receptor_preparation_summary_sha256"),
        ):
            verified(rooted(root, str(receptor[path_key])), str(receptor[hash_key]))
        for path_key, hash_key in (
            ("raw_sdf", "raw_sdf_sha256"),
            ("reference_sdf", "reference_sdf_sha256"),
            ("explicit_h_sdf", "explicit_h_sdf_sha256"),
            ("ligand_pdbqt", "ligand_pdbqt_sha256"),
            ("alignment_summary", "alignment_summary_sha256"),
        ):
            verified(rooted(root, str(case[path_key])), str(case[hash_key]))
        coordinates, _, _ = coordinates_and_elements_from_sdf_safe(
            rooted(root, str(case["reference_sdf"]))
        )
        return receptor, case, coordinates
    except (KeyError, OSError, TypeError, ValueError):
        return None


def prepare_case(
    selected: dict[str, str],
    config: dict[str, object],
    inputs: dict[str, Path],
    outputs: dict[str, Path],
    root: Path,
) -> tuple[dict[str, object], dict[str, object], np.ndarray]:
    conformer_id = selected["conformer_id"]
    pdb_id = selected["pdb_id"]
    ligand_resname = selected["selected_ligand_resname"]
    case_id = f"{pdb_id}_{ligand_resname}"
    case_root = outputs["run_directory"] / "preparation" / case_id
    case_root.mkdir(parents=True, exist_ok=True)
    raw_mmcif = verified(rooted(root, selected["mmcif_path"]), selected["mmcif_sha256"])
    aligned_pdb = verified(
        rooted(root, selected["aligned_protein_pdb_path"]),
        selected["aligned_protein_pdb_sha256"],
    )
    anchors = [int(value) for value in dict(config["reference"])["anchor_residue_numbers"]]
    alignment, raw_atoms, rotation, translation = alignment_audit_fa10(
        inputs["reference_mmcif"], raw_mmcif, selected["chain"], aligned_pdb, anchors
    )
    alignment["target"] = "BACE1"
    alignment["method"] = (
        "proper-rotation Kabsch alignment over all shared, residue-name-matched "
        "BACE1 C-alpha atoms"
    )
    alignment_path = case_root / "alignment_summary.json"
    write_json(alignment_path, alignment)

    receptor_root = case_root / "receptor"
    summary_path = receptor_root / "summary.json"
    protein_only = receptor_root / f"{conformer_id}_protein_only.pdb"
    prepared_pdb = receptor_root / f"{conformer_id}_prepared.pdb"
    receptor_pdbqt = receptor_root / f"{conformer_id}_receptor.pdbqt"
    command = [
            sys.executable,
            str(inputs["prepare_receptor_script"]),
            "--input-pdb",
            str(aligned_pdb),
            "--chain",
            "A",
            "--protein-only-output",
            str(protein_only),
            "--prepared-pdb-output",
            str(prepared_pdb),
            "--pdbqt-output",
            str(receptor_pdbqt),
            "--summary-output",
            str(summary_path),
            "--overwrite",
        ]
    override = dict(dict(config.get("receptor_preparation_overrides", {})).get(conformer_id, {}))
    if override:
        command.extend(["--set-template", str(override["set_template"])])
    run_checked(
        command,
        f"BACE1 receptor preparation for {conformer_id}",
    )
    receptor_evidence = read_json(summary_path)
    residue_change = dict(receptor_evidence["residue_count_change"])
    if (
        receptor_evidence.get("status") != "ok"
        or receptor_evidence.get("allow_bad_res") is not False
        or residue_change["input_protein_only"] != residue_change["output_pdbqt"]
    ):
        raise ValueError(f"BACE1 receptor preparation failed: {conformer_id}")

    download = dict(config["ligand_download"])
    source_url = build_modelserver_url(
        str(download["url_template"]),
        pdb_id,
        selected["chain"],
        int(selected["selected_ligand_resseq"]),
    )
    raw_sdf = case_root / f"{case_id}_raw.sdf"
    download_cached(
        source_url,
        raw_sdf,
        float(download["timeout_seconds"]),
        int(download["maximum_retries"]),
        float(download["retry_backoff_seconds"]),
    )
    raw_coordinates, raw_elements, raw_molecule = coordinates_and_elements_from_sdf_safe(raw_sdf)
    ligand_key = (
        ligand_resname,
        int(selected["selected_ligand_resseq"]),
        selected["selected_ligand_icode"],
    )
    raw_ligands = ligand_residue_map(raw_atoms, {ligand_resname})
    if ligand_key not in raw_ligands:
        raise ValueError(f"selected BACE1 mmCIF ligand is missing: {case_id}")
    ligand_atoms = raw_ligands[ligand_key]
    mmcif_coordinates = np.vstack([atom.coord for atom in ligand_atoms])
    mmcif_elements = [atom.element for atom in ligand_atoms]
    if raw_molecule.GetNumHeavyAtoms() != int(selected["selected_ligand_heavy_atom_count"]):
        raise ValueError(f"BACE1 ligand heavy-atom count differs: {case_id}")
    raw_rmsd, raw_maximum = point_set_rmsd_by_element(
        raw_coordinates, raw_elements, mmcif_coordinates, mmcif_elements
    )
    limit = float(dict(config["preparation_protocol"])["maximum_element_matched_coordinate_error_angstrom"])
    if raw_rmsd > limit or raw_maximum > limit:
        raise ValueError(f"BACE1 ModelServer ligand coordinates differ: {case_id}")

    common_sdf = case_root / f"{case_id}_common_frame.sdf"
    explicit_h_sdf = case_root / f"{case_id}_common_frame_explicitH.sdf"
    transform_sdf_safe(
        raw_sdf, common_sdf, case_root / "common_frame_transform.json", rotation, translation, False
    )
    transform_sdf_safe(
        raw_sdf,
        explicit_h_sdf,
        case_root / "explicit_h_transform.json",
        rotation,
        translation,
        True,
    )
    common_coordinates, common_elements, common_molecule = coordinates_and_elements_from_sdf_safe(common_sdf)
    transformed_mmcif = mmcif_coordinates @ rotation + translation
    common_rmsd, common_maximum = point_set_rmsd_by_element(
        common_coordinates, common_elements, transformed_mmcif, mmcif_elements
    )
    if common_rmsd > limit or common_maximum > limit:
        raise ValueError(f"BACE1 common-frame ligand coordinates differ: {case_id}")
    ligand_pdbqt = case_root / f"{case_id}_ligand.pdbqt"
    ligand_audit, ligand_variant = prepare_ligand_safe(explicit_h_sdf, ligand_pdbqt)

    receptor_row = {
        "conformer_id": conformer_id,
        "pdb_id": pdb_id,
        "chain": "A",
        "receptor_pdbqt": receptor_pdbqt.relative_to(root).as_posix(),
        "receptor_pdbqt_sha256": file_sha256(receptor_pdbqt),
        "receptor_preparation_summary": summary_path.relative_to(root).as_posix(),
        "receptor_preparation_summary_sha256": file_sha256(summary_path),
        "set_template_override": override.get("set_template", ""),
        "status": "ok",
    }
    case_row = {
        "case_id": case_id,
        "conformer_id": conformer_id,
        "pdb_id": pdb_id,
        "source_chain": selected["chain"],
        "selected_ligand_resname": ligand_key[0],
        "selected_ligand_resseq": ligand_key[1],
        "selected_ligand_icode": ligand_key[2],
        "selected_ligand_heavy_atom_count": common_molecule.GetNumHeavyAtoms(),
        "source_url": source_url,
        "raw_sdf": raw_sdf.relative_to(root).as_posix(),
        "raw_sdf_sha256": file_sha256(raw_sdf),
        "reference_sdf": common_sdf.relative_to(root).as_posix(),
        "reference_sdf_sha256": file_sha256(common_sdf),
        "explicit_h_sdf": explicit_h_sdf.relative_to(root).as_posix(),
        "explicit_h_sdf_sha256": file_sha256(explicit_h_sdf),
        "ligand_pdbqt": ligand_pdbqt.relative_to(root).as_posix(),
        "ligand_pdbqt_sha256": file_sha256(ligand_pdbqt),
        "ligand_preparation_variant": ligand_variant,
        "ligand_pdbqt_atom_count": ligand_audit["pdbqt_atom_count"],
        "raw_sdf_to_mmcif_element_matched_rmsd_angstrom": raw_rmsd,
        "raw_sdf_to_mmcif_maximum_atom_distance_angstrom": raw_maximum,
        "common_sdf_to_transformed_mmcif_element_matched_rmsd_angstrom": common_rmsd,
        "common_sdf_to_transformed_mmcif_maximum_atom_distance_angstrom": common_maximum,
        "alignment_summary": alignment_path.relative_to(root).as_posix(),
        "alignment_summary_sha256": file_sha256(alignment_path),
        "receptor_preparation_override": override.get("set_template", ""),
        "status": "ok",
    }
    write_json(
        case_root / "case_preparation_summary.json",
        {
            "schema_version": "1.0",
            "status": "stage41b_case_preparation_ok",
            "receptor_row": receptor_row,
            "case_row": case_row,
        },
    )
    return receptor_row, case_row, common_coordinates


def run(
    config_path: Path, root: Path, audit_only: bool, resume: bool
) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config, inputs, selected_rows, input_audit = validate_inputs(config_path, root)
    if audit_only:
        record = {
            "schema_version": "1.0",
            "experiment_id": config["experiment_id"],
            "operation": "input audit only; no download, preparation, or docking was started",
            **input_audit,
        }
        print(json.dumps(record, indent=2, sort_keys=True))
        return record

    load_preparation_dependencies()
    outputs = {key: rooted(root, str(value)) for key, value in dict(config["outputs"]).items()}
    receptor_rows: list[dict[str, object]] = []
    case_rows: list[dict[str, object]] = []
    coordinate_sets: list[np.ndarray] = []
    failure_rows: list[dict[str, object]] = []
    resumed_count = 0
    prepared_count = 0
    for index, selected in enumerate(selected_rows, start=1):
        case_id = f"{selected['pdb_id']}_{selected['selected_ligand_resname']}"
        checkpoint_path = outputs["run_directory"] / "preparation" / case_id / "case_preparation_summary.json"
        cached = checkpoint(root, checkpoint_path) if resume else None
        if cached is None:
            print(f"preparing {index}/{len(selected_rows)}: {selected['conformer_id']}", flush=True)
            try:
                receptor, case, coordinates = prepare_case(selected, config, inputs, outputs, root)
                prepared_count += 1
            except Exception as error:
                receptor_summary = checkpoint_path.parent / "receptor" / "summary.json"
                evidence = read_json(receptor_summary) if receptor_summary.is_file() else {}
                failure = {
                    "case_id": case_id,
                    "conformer_id": selected["conformer_id"],
                    "pdb_id": selected["pdb_id"],
                    "error": f"{type(error).__name__}: {error}",
                    "receptor_preparation_error": evidence.get("error", ""),
                    "meeko_return_code": evidence.get("meeko_return_code", ""),
                    "meeko_stderr": evidence.get("meeko_stderr", ""),
                    "status": "technical_preparation_failed",
                }
                write_json(checkpoint_path.parent / "case_preparation_failure.json", failure)
                failure_rows.append(failure)
                print(
                    f"preparation failed {index}/{len(selected_rows)}: "
                    f"{selected['conformer_id']}; continuing",
                    flush=True,
                )
                continue
        else:
            receptor, case, coordinates = cached
            resumed_count += 1
            print(f"resume ok {index}/{len(selected_rows)}: {selected['conformer_id']}", flush=True)
        receptor_rows.append(receptor)
        case_rows.append(case)
        coordinate_sets.append(coordinates)

    box_rule = dict(config["common_box_rule"])
    box = derive_common_box(
        coordinate_sets,
        float(box_rule["minimum_crystal_pose_margin_angstrom"]),
        float(box_rule["size_increment_angstrom"]),
        float(box_rule["minimum_axis_size_angstrom"]),
        int(box_rule["center_decimal_places"]),
    )
    if max(box["size"].values()) > float(box_rule["maximum_axis_size_angstrom"]):
        raise ValueError("BACE1 large-pool common box is too large")
    box_record = {
        "schema_version": "1.0",
        "status": "stage41b_bace1_large_pool_common_box_ok",
        "derivation_rule": box_rule,
        "receptor_count": len(receptor_rows),
        **box,
    }
    write_csv(outputs["receptor_manifest_csv"], receptor_rows)
    write_csv(outputs["redocking_case_manifest_csv"], case_rows)
    write_json(outputs["common_box_json"], box_record)
    minimum_prepared = int(dict(config["expected"])["minimum_prepared_receptor_count"])
    gate_ready = len(receptor_rows) >= minimum_prepared
    if len(receptor_rows) == len(selected_rows):
        status = "stage41b_bace1_large_pool_redocking_inputs_ok"
    elif gate_ready:
        status = "stage41b_bace1_large_pool_redocking_inputs_partial_gate_ready"
    else:
        status = "stage41b_bace1_large_pool_redocking_inputs_failed"
    result = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": status,
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "counts": {
            "frozen_receptor_count": len(selected_rows),
            "prepared_receptor_count": len(receptor_rows),
            "prepared_cognate_ligand_count": len(case_rows),
            "prepared_this_invocation": prepared_count,
            "resumed_this_invocation": resumed_count,
            "failed_case_count": len(failure_rows),
            "minimum_prepared_receptor_count": minimum_prepared,
        },
        "technical_gate_ready": gate_ready,
        "failed_cases": failure_rows,
        "common_box": box_record,
        "maximum_raw_ligand_coordinate_rmsd_angstrom": max(
            float(row["raw_sdf_to_mmcif_element_matched_rmsd_angstrom"])
            for row in case_rows
        ),
        "maximum_common_ligand_coordinate_rmsd_angstrom": max(
            float(row["common_sdf_to_transformed_mmcif_element_matched_rmsd_angstrom"])
            for row in case_rows
        ),
        "data_boundary": input_audit["data_boundary"],
        "outputs": {
            key: {
                "path": outputs[key].relative_to(root).as_posix(),
                "sha256": file_sha256(outputs[key]),
            }
            for key in (
                "receptor_manifest_csv",
                "redocking_case_manifest_csv",
                "common_box_json",
            )
        },
        "next_gate": (
            "run three-seed BACE1 Uni-Dock cognate redocking for every successfully prepared frozen receptor; technical preparation failures count as receptor failures and cannot be replaced"
            if gate_ready
            else "stop the BACE1 large-pool route because fewer than 40 frozen receptors were preparation-ready"
        ),
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["summary_json"], result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run(args.config, args.root, args.audit_only, args.resume)
    return 1 if str(result.get("status", "")).endswith("_failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())
