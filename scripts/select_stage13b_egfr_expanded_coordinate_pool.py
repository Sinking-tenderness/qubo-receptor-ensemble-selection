"""Expand the frozen EGFR metadata pool and rerun structural selection."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import gemmi
import numpy as np

try:
    from .select_stage13_egfr_coordinate_pool import (
        audit_structure,
        ca_map,
        derive_reference_residues,
        download_one,
        file_sha256,
        ligand_residue_map,
        load_config,
        maxmin_select,
        read_csv,
        read_json,
        select_chain_atoms,
        verified,
        write_csv,
        write_json,
    )
except ImportError:
    from select_stage13_egfr_coordinate_pool import (
        audit_structure,
        ca_map,
        derive_reference_residues,
        download_one,
        file_sha256,
        ligand_residue_map,
        load_config,
        maxmin_select,
        read_csv,
        read_json,
        select_chain_atoms,
        verified,
        write_csv,
        write_json,
    )


def select_expanded_metadata_candidates(
    rows: list[dict[str, str]], amendment: dict[str, object]
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    expansion = dict(amendment["metadata_pool_expansion"])
    maximum = float(expansion["expanded_maximum_resolution_angstrom"])
    original = [row for row in rows if row["status"] == "metadata_eligible"]
    newly_admitted = [
        row
        for row in rows
        if row["exclusion_reasons"] == "resolution_above_limit"
        and float(row["resolution_angstrom"]) <= maximum
    ]
    newly_admitted.sort(key=lambda row: row["pdb_id"])
    expected_ids = [str(value) for value in expansion["newly_admitted_pdb_ids"]]
    if [row["pdb_id"] for row in newly_admitted] != expected_ids:
        raise ValueError("Stage 13b newly admitted metadata IDs differ")
    if len(original) != int(expansion["original_metadata_eligible_count"]):
        raise ValueError("Stage 13b original metadata count differs")
    expanded = sorted(original + newly_admitted, key=lambda row: row["pdb_id"])
    if len(expanded) != int(expansion["expanded_metadata_candidate_count"]):
        raise ValueError("Stage 13b expanded metadata count differs")
    return expanded, newly_admitted


def write_report(path: Path, summary: dict[str, object]) -> None:
    counts = dict(summary["counts"])
    lines = [
        "# Stage 13b EGFR Expanded Coordinate Pool",
        "",
        "## Result",
        "",
        f"- Original coordinate-eligible structures: {counts['reused_original_eligible_count']}",
        f"- Newly audited structures: {counts['newly_audited_count']}",
        f"- Combined coordinate-eligible structures: {counts['combined_coordinate_eligible_count']}",
        f"- Selected receptors: {counts['selected_receptor_count']}",
        f"- Status: {summary['status']}",
        "",
        "The original coordinate failures were retained without readmission or re-audit.",
        "No activity label, docking score, fresh-validation row, or test row was read.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def run(config_path: Path, overwrite: bool = False) -> dict[str, object]:
    config = load_config(config_path)
    implementation = dict(config["implementation"])
    script_path = verified(
        Path(str(implementation["path"])), str(implementation["sha256"])
    )
    if script_path.resolve() != Path(__file__).resolve():
        raise ValueError("Stage 13b implementation path differs")
    verified(
        Path(str(implementation["base_dependency_path"])),
        str(implementation["base_dependency_sha256"]),
    )
    verified(
        Path(str(implementation["math_dependency_path"])),
        str(implementation["math_dependency_sha256"]),
    )
    prereg_record = dict(config["preregistration"])
    coordinate_amendment_record = dict(config["preregistration_amendment"])
    expansion_record = dict(config["pool_expansion_amendment"])
    prereg_path = verified(
        Path(str(prereg_record["path"])), str(prereg_record["sha256"])
    )
    coordinate_amendment_path = verified(
        Path(str(coordinate_amendment_record["path"])),
        str(coordinate_amendment_record["sha256"]),
    )
    expansion_path = verified(
        Path(str(expansion_record["path"])), str(expansion_record["sha256"])
    )
    coordinate_amendment = read_json(coordinate_amendment_path)
    expansion_amendment = read_json(expansion_path)
    if expansion_amendment["coordinate_gate_amendment"]["sha256"] != file_sha256(
        coordinate_amendment_path
    ):
        raise ValueError("Stage 13b coordinate amendment differs")
    if any(
        int(value) != 0
        for value in dict(expansion_amendment["data_boundary"]).values()
    ):
        raise ValueError("Stage 13b data boundary differs")

    expected_runtime = {
        key: str(value) for key, value in dict(config["runtime"]).items()
    }
    runtime = {
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "gemmi_version": gemmi.__version__,
    }
    if runtime != expected_runtime:
        raise RuntimeError(f"Stage 13b runtime differs: {runtime} != {expected_runtime}")

    inputs: dict[str, Path] = {}
    for key, value in dict(config["inputs"]).items():
        record = dict(value)
        inputs[key] = verified(Path(str(record["path"])), str(record["sha256"]))
    original_summary = read_json(inputs["original_coordinate_summary"])
    trigger = dict(expansion_amendment["trigger"])
    if (
        original_summary.get("status") != trigger["required_status"]
        or int(original_summary["counts"]["coordinate_eligible_count"])
        != int(trigger["observed_coordinate_eligible_count"])
    ):
        raise ValueError("Stage 13b trigger summary differs")
    if any(int(value) != 0 for value in original_summary["data_boundary"].values()):
        raise ValueError("Stage 13b upstream data boundary differs")

    outputs = {key: Path(str(value)) for key, value in dict(config["outputs"]).items()}
    directory_keys = {"raw_mmcif_directory", "aligned_protein_pdb_directory"}
    existing = [
        path
        for key, path in outputs.items()
        if key not in directory_keys and path.exists()
    ]
    if existing and not overwrite:
        raise FileExistsError("Stage 13b outputs exist; pass --overwrite")
    if overwrite:
        for path in existing:
            path.unlink()

    metadata_rows = read_csv(inputs["candidate_metadata_csv"])
    expanded_candidates, new_candidates = select_expanded_metadata_candidates(
        metadata_rows, expansion_amendment
    )
    original_audit = read_csv(inputs["original_coordinate_audit_csv"])
    original_eligible = [
        row for row in original_audit if row["status"] == "coordinate_eligible"
    ]
    if len(original_eligible) != int(trigger["observed_coordinate_eligible_count"]):
        raise ValueError("Stage 13b original eligible audit count differs")
    original_feature_rows = read_csv(inputs["original_feature_matrix_csv"])
    original_feature_by_id = {
        row["conformer_id"]: row for row in original_feature_rows
    }
    if set(original_feature_by_id) != {
        row["conformer_id"] for row in original_eligible
    }:
        raise ValueError("Stage 13b original feature IDs differ")
    feature_names = [
        name for name in original_feature_rows[0] if name != "conformer_id"
    ]
    vectors = {
        conformer_id: np.array(
            [float(row[name]) for name in feature_names], dtype=float
        )
        for conformer_id, row in original_feature_by_id.items()
    }

    raw_directory = outputs["raw_mmcif_directory"]
    raw_paths = {
        row["pdb_id"]: raw_directory / f"{row['pdb_id']}.cif"
        for row in new_candidates
    }
    download = dict(config["download"])
    with ThreadPoolExecutor(max_workers=int(download["workers"])) as executor:
        futures = {
            executor.submit(
                download_one,
                pdb_id,
                path,
                str(download["url_template"]),
                float(download["timeout_seconds"]),
                int(download["maximum_retries"]),
                float(download["retry_backoff_seconds"]),
            ): pdb_id
            for pdb_id, path in raw_paths.items()
        }
        download_errors: dict[str, str] = {}
        for future in as_completed(futures):
            pdb_id, error = future.result()
            if error:
                download_errors[pdb_id] = error

    reference_record = dict(coordinate_amendment["reference"])
    reference_structure = gemmi.read_structure(str(inputs["reference_mmcif"]))
    reference_atoms = select_chain_atoms(
        reference_structure, str(reference_record["auth_chain"])
    )
    reference_ligand_id = str(reference_record["ligand_comp_id"])
    pocket_numbers = [
        int(value) for value in reference_record["reference_pocket_residue_numbers"]
    ]
    anchor_numbers = {
        int(value) for value in reference_record["required_anchor_residue_numbers"]
    }
    if derive_reference_residues(reference_atoms, reference_ligand_id, 6.0) != pocket_numbers:
        raise ValueError("Stage 13b reference pocket reconstruction differs")
    if derive_reference_residues(reference_atoms, reference_ligand_id, 4.0) != sorted(anchor_numbers):
        raise ValueError("Stage 13b reference anchor reconstruction differs")
    if len(ca_map(reference_atoms)) != int(reference_record["visible_protein_ca_count"]):
        raise ValueError("Stage 13b reference C-alpha count differs")
    reference_ligands = ligand_residue_map(reference_atoms, {reference_ligand_id})
    reference_ligand_coords = np.vstack(
        [atom.coord for atom in next(iter(reference_ligands.values()))]
    )
    gate = dict(coordinate_amendment["coordinate_gate"])

    new_audit: list[dict[str, object]] = []
    new_eligible: list[dict[str, object]] = []
    aligned_directory = outputs["aligned_protein_pdb_directory"]
    for metadata in new_candidates:
        pdb_id = metadata["pdb_id"]
        conformer_id = f"EGFR_{pdb_id}_aligned"
        if pdb_id in download_errors:
            row = {
                "conformer_id": conformer_id,
                "pdb_id": pdb_id,
                "chain": metadata["selected_auth_chain"],
                "status": "coordinate_excluded",
                "exclusion_reasons": "coordinate_file_unavailable",
                "download_error": download_errors[pdb_id],
                "mmcif_path": "",
                "mmcif_sha256": "",
                "aligned_protein_pdb_path": "",
                "aligned_protein_pdb_sha256": "",
            }
            vector = None
            names = None
        else:
            try:
                row, vector, names = audit_structure(
                    pdb_id,
                    metadata["selected_auth_chain"],
                    {
                        value
                        for value in metadata["qualifying_ligand_ids"].split(";")
                        if value
                    },
                    raw_paths[pdb_id],
                    aligned_directory / f"{conformer_id}_to_2RGP_A.pdb",
                    reference_atoms,
                    reference_ligand_coords,
                    pocket_numbers,
                    anchor_numbers,
                    gate,
                )
            except Exception as error:
                row, vector, names = (
                    {
                        "conformer_id": conformer_id,
                        "pdb_id": pdb_id,
                        "chain": metadata["selected_auth_chain"],
                        "status": "coordinate_excluded",
                        "exclusion_reasons": "coordinate_parse_or_audit_error",
                        "audit_error": f"{type(error).__name__}: {error}",
                        "mmcif_path": raw_paths[pdb_id].as_posix(),
                        "mmcif_sha256": file_sha256(raw_paths[pdb_id]),
                        "aligned_protein_pdb_path": "",
                        "aligned_protein_pdb_sha256": "",
                    },
                    None,
                    None,
                )
        new_audit.append(row)
        if vector is not None:
            if names != feature_names:
                raise ValueError("Stage 13b new feature names differ")
            vectors[conformer_id] = vector
            new_eligible.append(row)

    combined_audit: list[dict[str, object]] = [dict(row) for row in original_audit]
    combined_audit.extend(new_audit)
    combined_audit.sort(key=lambda row: str(row["conformer_id"]))
    combined_eligible: list[dict[str, object]] = [dict(row) for row in original_eligible]
    combined_eligible.extend(new_eligible)
    combined_eligible.sort(key=lambda row: str(row["conformer_id"]))
    eligible_by_id = {str(row["conformer_id"]): row for row in combined_eligible}
    if len(combined_audit) != len(expanded_candidates):
        raise ValueError("Stage 13b combined audit count differs")

    ordered_ids = sorted(vectors)
    matrix = np.vstack([vectors[conformer_id] for conformer_id in ordered_ids])
    standard_deviations = matrix.std(axis=0)
    keep = standard_deviations >= float(
        coordinate_amendment["structural_selection"]["minimum_variable_feature_sd_angstrom"]
    )
    variable_feature_count = int(keep.sum())
    if variable_feature_count < 3:
        raise ValueError("too few variable Stage 13b structural features")
    means = matrix.mean(axis=0)
    standardized = (matrix[:, keep] - means[keep]) / standard_deviations[keep]
    standardized /= math.sqrt(variable_feature_count)
    distance_by_pair: dict[tuple[str, str], float] = {}
    distance_rows: list[dict[str, object]] = []
    for first_index, second_index in combinations(range(len(ordered_ids)), 2):
        first = ordered_ids[first_index]
        second = ordered_ids[second_index]
        distance = float(
            np.linalg.norm(standardized[first_index] - standardized[second_index])
        )
        distance_by_pair[(first, second)] = distance
        distance_rows.append(
            {
                "conformer_id_a": first,
                "conformer_id_b": second,
                "standardized_pocket_distance": distance,
            }
        )
    target_count = int(
        coordinate_amendment["structural_selection"]["target_receptor_count"]
    )
    selected_rows: list[dict[str, object]] = []
    reference_id = "EGFR_2RGP_reference"
    if len(ordered_ids) >= target_count:
        additions = maxmin_select(
            ordered_ids,
            [reference_id],
            distance_by_pair,
            target_count - 1,
        )
        selected_rows.append(
            {
                "pool_role": "reference_seed",
                "selection_rank": 1,
                "minimum_standardized_distance_to_selected_pool": "",
                **eligible_by_id[reference_id],
            }
        )
        for addition in additions:
            conformer_id = str(addition["conformer_id"])
            selected_rows.append(
                {
                    "pool_role": "maxmin_addition",
                    "selection_rank": int(addition["selection_rank"]) + 1,
                    "minimum_standardized_distance_to_selected_pool": addition[
                        "minimum_standardized_distance_to_selected_pool"
                    ],
                    **eligible_by_id[conformer_id],
                }
            )

    feature_rows = [
        {
            "conformer_id": conformer_id,
            **{
                name: float(value)
                for name, value in zip(feature_names, vectors[conformer_id])
            },
        }
        for conformer_id in ordered_ids
    ]
    eligible_manifest = [
        {
            "conformer_id": row["conformer_id"],
            "pdb_id": row["pdb_id"],
            "chain": row["chain"],
            "mmcif_path": row["mmcif_path"],
            "mmcif_sha256": row["mmcif_sha256"],
            "aligned_protein_pdb_path": row["aligned_protein_pdb_path"],
            "aligned_protein_pdb_sha256": row["aligned_protein_pdb_sha256"],
            "selected_ligand_resname": row["selected_ligand_resname"],
            "selected_ligand_resseq": row["selected_ligand_resseq"],
            "selected_ligand_icode": row["selected_ligand_icode"],
            "source_pool": (
                "original_coordinate_eligible"
                if row["conformer_id"] in original_feature_by_id
                else "resolution_expansion_newly_eligible"
            ),
        }
        for row in combined_eligible
    ]
    write_csv(outputs["combined_coordinate_audit_csv"], combined_audit)
    write_csv(outputs["combined_eligible_pool_manifest_csv"], eligible_manifest)
    write_csv(outputs["combined_feature_matrix_csv"], feature_rows)
    write_csv(outputs["combined_pairwise_distances_csv"], distance_rows)
    if selected_rows:
        write_csv(outputs["selected16_manifest_csv"], selected_rows)

    reasons = Counter(
        reason
        for row in combined_audit
        for reason in str(row.get("exclusion_reasons", "")).split(";")
        if reason
    )
    hidden_covalent_count = sum(
        "explicit_target_ligand_covalent_connection" in str(row.get("exclusion_reasons", ""))
        or "protein_ligand_distance_indicates_covalency" in str(row.get("exclusion_reasons", ""))
        for row in combined_audit
    )
    status = (
        "stage13b_egfr_expanded_structural_selection_ok"
        if len(selected_rows) == target_count
        else "stage13b_egfr_expanded_coordinate_pool_insufficient_stop"
    )
    output_records = {}
    for key, path in outputs.items():
        if key in directory_keys or key in {"summary_json", "report_md"} or not path.is_file():
            continue
        output_records[key] = {"path": path.as_posix(), "sha256": file_sha256(path)}
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": status,
        "config": {"path": config_path.as_posix(), "sha256": file_sha256(config_path)},
        "preregistration": {"path": prereg_path.as_posix(), "sha256": file_sha256(prereg_path)},
        "coordinate_gate_amendment": {"path": coordinate_amendment_path.as_posix(), "sha256": file_sha256(coordinate_amendment_path)},
        "pool_expansion_amendment": {"path": expansion_path.as_posix(), "sha256": file_sha256(expansion_path)},
        "runtime": runtime,
        "counts": {
            "expanded_metadata_candidate_count": len(expanded_candidates),
            "reused_original_audit_count": len(original_audit),
            "reused_original_eligible_count": len(original_eligible),
            "newly_audited_count": len(new_audit),
            "newly_coordinate_eligible_count": len(new_eligible),
            "combined_coordinate_eligible_count": len(combined_eligible),
            "combined_coordinate_excluded_count": len(combined_audit) - len(combined_eligible),
            "hidden_covalent_exclusion_count": hidden_covalent_count,
            "selected_receptor_count": len(selected_rows),
            "target_receptor_count": target_count,
            "raw_feature_count": len(feature_names),
            "variable_feature_count": variable_feature_count,
            "pairwise_distance_count": len(distance_rows),
        },
        "coordinate_exclusion_reason_counts": dict(sorted(reasons.items())),
        "selected_receptor_ids": [str(row["conformer_id"]) for row in selected_rows],
        "data_boundary": {
            "ligand_labels_read": 0,
            "docking_scores_read": 0,
            "MAPK14_stage11_rows_read": 0,
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
        },
        "outputs": output_records,
        "next_gate": (
            "freeze deterministic heavy-atom completion, receptor and native-ligand preparation, then require co-crystal redocking RMSD at or below 2.0 A"
            if status == "stage13b_egfr_expanded_structural_selection_ok"
            else "stop without docking and redesign the receptor-pool strategy prospectively"
        ),
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["summary_json"], summary)
    write_report(outputs["report_md"], summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
