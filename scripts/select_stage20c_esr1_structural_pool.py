"""Download, audit, and select a label-independent ESR1 structural pool."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import gemmi
import numpy as np

try:
    from scripts.select_mk14_rcsb_coordinate_pool import maxmin_select
    from scripts.select_stage13_egfr_coordinate_pool import (
        audit_structure,
        ca_map,
        derive_reference_residues,
        download_one,
        file_sha256,
        ligand_residue_map,
        read_csv,
        read_json,
        select_chain_atoms,
        write_csv,
        write_json,
    )
except ModuleNotFoundError:
    from select_mk14_rcsb_coordinate_pool import maxmin_select
    from select_stage13_egfr_coordinate_pool import (
        audit_structure,
        ca_map,
        derive_reference_residues,
        download_one,
        file_sha256,
        ligand_residue_map,
        read_csv,
        read_json,
        select_chain_atoms,
        write_csv,
        write_json,
    )


def verified(root: Path, record: dict[str, object]) -> Path:
    path = root / str(record["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if file_sha256(path) != str(record["sha256"]).upper():
        raise ValueError(f"SHA-256 differs: {path}")
    return path


def conformer_id(pdb_id: str) -> str:
    return "ESR1_1SJ0_reference" if pdb_id == "1SJ0" else f"ESR1_{pdb_id}_aligned"


def write_report(path: Path, summary: dict[str, object]) -> None:
    counts = dict(summary["counts"])
    lines = [
        "# Stage 20c ESR1 Structural Selection",
        "",
        f"- Metadata candidates audited: {counts['audited_count']}",
        f"- Coordinate eligible: {counts['coordinate_eligible_count']}",
        f"- Selected for redocking: {counts['selected_receptor_count']}",
        f"- Status: {summary['status']}",
        "",
        "Selection used coordinates and chemical-component metadata only.",
        "No ESR1 docking score, validation row, test row, or PPARG Stage 19c enrichment row was read.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    if file_sha256(Path(__file__)) != str(config["implementation"]["sha256"]).upper():
        raise ValueError("Stage 20c implementation SHA-256 differs")
    preregistration_path = verified(root, dict(config["preregistration"]))
    preregistration = read_json(preregistration_path)
    discovery_path = verified(root, dict(config["discovery_summary"]))
    discovery = read_json(discovery_path)
    if discovery["status"] != "stage20b_esr1_metadata_discovery_ok":
        raise ValueError("Stage 20b ESR1 metadata discovery did not pass")
    if any(int(value) != 0 for value in dict(discovery["data_boundary"]).values()):
        raise ValueError("Stage 20b crossed a protected data boundary")
    boundary = preregistration["data_boundary"]
    if any(
        int(boundary[key]) != 0
        for key in (
            "ESR1_benchmark_docking_scores_read",
            "ESR1_fresh_validation_rows_read",
            "ESR1_test_rows_read",
            "PPARG_stage19c_enrichment_rows_read",
        )
    ):
        raise ValueError("Stage 20c crossed a protected data boundary")

    metadata_path = verified(root, dict(config["inputs"]["eligible_metadata_csv"]))
    reference_path = verified(root, dict(config["inputs"]["reference_mmcif"]))
    candidates = read_csv(metadata_path)
    if len(candidates) != int(discovery["counts"]["metadata_eligible_count"]):
        raise ValueError("ESR1 metadata-candidate count differs")
    if len({row["pdb_id"] for row in candidates}) != len(candidates):
        raise ValueError("ESR1 metadata candidates contain duplicate PDB IDs")

    outputs = {key: root / str(value) for key, value in dict(config["outputs"]).items()}
    protected_outputs = [
        path
        for key, path in outputs.items()
        if key not in {"raw_mmcif_directory", "aligned_protein_pdb_directory"}
    ]
    if any(path.exists() for path in protected_outputs) and not overwrite:
        raise FileExistsError("Stage 20c outputs exist; pass --overwrite")

    raw_directory = outputs["raw_mmcif_directory"]
    raw_paths = {
        row["pdb_id"]: (
            reference_path
            if row["pdb_id"] == "1SJ0"
            else raw_directory / f"{row['pdb_id']}.cif"
        )
        for row in candidates
    }
    download = dict(config["download"])
    errors: dict[str, str] = {}
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
        for future in as_completed(futures):
            pdb_id, error = future.result()
            if error:
                errors[pdb_id] = error

    reference = dict(config["reference"])
    structure = gemmi.read_structure(str(reference_path))
    reference_atoms = select_chain_atoms(structure, str(reference["auth_chain"]))
    ligand_id = str(reference["ligand_comp_id"])
    pocket_numbers = [int(value) for value in reference["pocket_residue_numbers"]]
    anchor_numbers = {int(value) for value in reference["anchor_residue_numbers"]}
    if derive_reference_residues(reference_atoms, ligand_id, 6.0) != pocket_numbers:
        raise ValueError("ESR1 reference pocket reconstruction differs")
    if derive_reference_residues(reference_atoms, ligand_id, 4.0) != sorted(anchor_numbers):
        raise ValueError("ESR1 reference anchor reconstruction differs")
    if len(ca_map(reference_atoms)) != int(reference["visible_protein_ca_count"]):
        raise ValueError("ESR1 reference C-alpha count differs")
    reference_ligands = ligand_residue_map(reference_atoms, {ligand_id})
    if len(reference_ligands) != 1:
        raise ValueError("ESR1 reference ligand is ambiguous")
    reference_ligand_coords = np.vstack(
        [atom.coord for atom in next(iter(reference_ligands.values()))]
    )

    gate = dict(config["coordinate_gate"])
    vectors: dict[str, np.ndarray] = {}
    feature_names: list[str] | None = None
    audit_rows: list[dict[str, object]] = []
    aligned_directory = outputs["aligned_protein_pdb_directory"]
    for metadata in sorted(candidates, key=lambda row: row["pdb_id"]):
        pdb_id = metadata["pdb_id"]
        candidate_id = conformer_id(pdb_id)
        if pdb_id in errors:
            audit_rows.append(
                {
                    "conformer_id": candidate_id,
                    "pdb_id": pdb_id,
                    "chain": metadata["selected_auth_chain"],
                    "status": "coordinate_excluded",
                    "exclusion_reasons": "coordinate_file_unavailable",
                    "download_error": errors[pdb_id],
                }
            )
            continue
        aligned_path = aligned_directory / f"{candidate_id}_to_1SJ0_A.pdb"
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
                aligned_path,
                reference_atoms,
                reference_ligand_coords,
                pocket_numbers,
                anchor_numbers,
                gate,
            )
            row["conformer_id"] = candidate_id
            row["title"] = metadata["title"]
            row["resolution_angstrom"] = metadata["resolution_angstrom"]
            row["qualifying_ligand_ids"] = metadata["qualifying_ligand_ids"]
        except Exception as error:
            row, vector, names = (
                {
                    "conformer_id": candidate_id,
                    "pdb_id": pdb_id,
                    "chain": metadata["selected_auth_chain"],
                    "status": "coordinate_excluded",
                    "exclusion_reasons": "coordinate_parse_or_audit_error",
                    "audit_error": f"{type(error).__name__}: {error}",
                    "mmcif_path": raw_paths[pdb_id].relative_to(root).as_posix(),
                    "mmcif_sha256": file_sha256(raw_paths[pdb_id]),
                },
                None,
                None,
            )
        audit_rows.append(row)
        if row["status"] == "coordinate_eligible":
            if vector is None or names is None:
                raise ValueError("ESR1 eligible candidate has no feature vector")
            vectors[candidate_id] = vector
            if feature_names is None:
                feature_names = names
            elif feature_names != names:
                raise ValueError("ESR1 structural feature names differ")

    eligible_rows = [row for row in audit_rows if row["status"] == "coordinate_eligible"]
    reference_id = "ESR1_1SJ0_reference"
    if reference_id not in vectors:
        raise ValueError("ESR1 reference failed coordinate eligibility")
    target_count = int(config["structural_selection"]["target_receptor_count"])
    status = "stage20c_esr1_coordinate_pool_insufficient_stop"
    selected_rows: list[dict[str, object]] = []
    feature_rows: list[dict[str, object]] = []
    distance_rows: list[dict[str, object]] = []
    variable_feature_count = 0
    if len(vectors) >= target_count:
        ordered_ids = sorted(vectors)
        matrix = np.vstack([vectors[value] for value in ordered_ids])
        deviations = matrix.std(axis=0)
        keep = deviations >= float(
            config["structural_selection"]["minimum_variable_feature_sd_angstrom"]
        )
        variable_feature_count = int(keep.sum())
        if variable_feature_count < 3:
            raise ValueError("too few variable ESR1 structural features")
        standardized = (matrix[:, keep] - matrix[:, keep].mean(axis=0)) / deviations[keep]
        standardized /= math.sqrt(variable_feature_count)
        distances: dict[tuple[str, str], float] = {}
        for first, second in combinations(range(len(ordered_ids)), 2):
            distance = float(np.linalg.norm(standardized[first] - standardized[second]))
            pair = (ordered_ids[first], ordered_ids[second])
            distances[pair] = distance
            distance_rows.append(
                {
                    "conformer_id_a": pair[0],
                    "conformer_id_b": pair[1],
                    "standardized_pocket_distance": distance,
                }
            )
        additions = maxmin_select(
            ordered_ids, [reference_id], distances, target_count - 1
        )
        eligible_by_id = {str(row["conformer_id"]): row for row in eligible_rows}
        selected_rows.append(
            {
                "pool_role": "reference_seed",
                "selection_rank": 1,
                "minimum_standardized_distance_to_selected_pool": "",
                **eligible_by_id[reference_id],
            }
        )
        for addition in additions:
            selected_rows.append(
                {
                    "pool_role": "maxmin_addition",
                    "selection_rank": int(addition["selection_rank"]) + 1,
                    "minimum_standardized_distance_to_selected_pool": addition[
                        "minimum_standardized_distance_to_selected_pool"
                    ],
                    **eligible_by_id[str(addition["conformer_id"])],
                }
            )
        feature_rows = [
            {
                "conformer_id": value,
                **{
                    name: float(number)
                    for name, number in zip(feature_names or [], vectors[value])
                },
            }
            for value in ordered_ids
        ]
        status = "stage20c_esr1_structural_pool_ok"

    write_csv(outputs["coordinate_audit_csv"], audit_rows)
    write_csv(outputs["eligible_pool_manifest_csv"], eligible_rows)
    if feature_rows:
        write_csv(outputs["feature_matrix_csv"], feature_rows)
        write_csv(outputs["pairwise_distances_csv"], distance_rows)
        write_csv(outputs["selected24_manifest_csv"], selected_rows)
    reasons = Counter(
        reason
        for row in audit_rows
        for reason in str(row.get("exclusion_reasons", "")).split(";")
        if reason
    )
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": status,
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "counts": {
            "audited_count": len(audit_rows),
            "download_failure_count": len(errors),
            "coordinate_eligible_count": len(eligible_rows),
            "coordinate_excluded_count": len(audit_rows) - len(eligible_rows),
            "preparation_ready_coordinate_eligible_count": sum(
                int(row["global_incomplete_standard_amino_acid_residue_count"]) == 0
                for row in eligible_rows
            ),
            "selected_receptor_count": len(selected_rows),
            "target_receptor_count": target_count,
            "raw_feature_count": len(feature_names or []),
            "variable_feature_count": variable_feature_count,
            "pairwise_distance_count": len(distance_rows),
        },
        "coordinate_exclusion_reason_counts": dict(sorted(reasons.items())),
        "reference": {
            "conformer_id": reference_id,
            "pocket_residue_numbers": pocket_numbers,
            "anchor_residue_numbers": sorted(anchor_numbers),
        },
        "selected_receptor_ids": [str(row["conformer_id"]) for row in selected_rows],
        "data_boundary": {
            "ligand_labels_read": 0,
            "docking_scores_read": 0,
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
            "pparg_stage19c_enrichment_rows_read": 0,
        },
        "next_gate": (
            "apply the frozen preparation-readiness audit, then prepare ESR1 receptors and cognate ligands"
            if status.endswith("_ok")
            else "do not dock; close the ESR1 coordinate-pool gate"
        ),
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["summary_json"], summary)
    write_report(outputs["report_md"], summary)
    print(json.dumps(summary, indent=2, ensure_ascii=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    run(args.config, args.root, args.overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
