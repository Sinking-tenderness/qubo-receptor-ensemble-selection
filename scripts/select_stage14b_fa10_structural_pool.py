"""Download, audit, and select the frozen FA10 16+1 structural pool."""

from __future__ import annotations

import argparse
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
    from .select_mk14_rcsb_coordinate_pool import maxmin_select
    from .select_stage13_egfr_coordinate_pool import (
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
except ImportError:
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


def fa10_conformer_id(pdb_id: str) -> str:
    return "FA10_3KL6_reference" if pdb_id == "3KL6" else f"FA10_{pdb_id}_aligned"


def normalize_fa10_alignment(path: Path) -> None:
    text = path.read_text(encoding="ascii")
    old = "REMARK 900 LABEL-INDEPENDENT EGFR STRUCTURAL POOL ALIGNMENT"
    new = "REMARK 900 LABEL-INDEPENDENT FA10 STRUCTURAL POOL ALIGNMENT"
    if old not in text:
        raise ValueError(f"aligned PDB provenance remark differs: {path}")
    path.write_text(text.replace(old, new, 1), encoding="ascii")


def write_report(path: Path, summary: dict[str, object]) -> None:
    counts = dict(summary["counts"])
    lines = [
        "# Stage 14b FA10 Structural Pool",
        "",
        "## Result",
        "",
        f"- Metadata candidates audited: {counts['audited_count']}",
        f"- Coordinate eligible: {counts['coordinate_eligible_count']}",
        f"- Preparation ready: {counts['preparation_ready_count']}",
        f"- Selected receptors: {counts['selected_receptor_count']}",
        f"- Frozen reserve receptors: {counts['reserve_receptor_count']}",
        "",
        "Selection used coordinates only. No FA10 benchmark score or held-out row was read.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="ascii")


def run(config_path: Path, root: Path, overwrite: bool) -> dict[str, object]:
    root = root.resolve()
    config_path = config_path.resolve()
    config = read_json(config_path)
    implementation = dict(config["implementation"])
    if file_sha256(Path(__file__)) != str(implementation["sha256"]).upper():
        raise ValueError("Stage 14b implementation SHA-256 differs")
    for dependency in implementation["dependencies"]:
        verified(root, dict(dependency))

    preregistration = read_json(verified(root, dict(config["preregistration"])))
    discovery = read_json(verified(root, dict(config["discovery_summary"])))
    if discovery["status"] != "stage14a_fa10_metadata_discovery_ok":
        raise ValueError("Stage 14a FA10 metadata discovery did not pass")
    if any(int(value) != 0 for value in dict(discovery["data_boundary"]).values()):
        raise ValueError("Stage 14a crossed a protected data boundary")

    expected_runtime = {
        key: str(value) for key, value in dict(config["runtime"]).items()
    }
    runtime = {
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV", ""),
        "python_version": platform.python_version(),
        "gemmi_version": gemmi.__version__,
        "numpy_version": np.__version__,
    }
    if runtime != expected_runtime:
        raise RuntimeError(f"Stage 14b runtime differs: {runtime} != {expected_runtime}")

    inputs = dict(config["inputs"])
    metadata_path = verified(root, dict(inputs["eligible_metadata_csv"]))
    reference_path = verified(root, dict(inputs["reference_mmcif"]))
    metadata_rows = read_csv(metadata_path)
    if len(metadata_rows) != int(discovery["counts"]["metadata_eligible_count"]):
        raise ValueError("Stage 14b metadata row count differs")

    outputs = {
        key: root / str(value) for key, value in dict(config["outputs"]).items()
    }
    protected_outputs = [
        path for key, path in outputs.items() if key not in {"raw_mmcif_directory", "aligned_directory"}
    ]
    if any(path.exists() for path in protected_outputs) and not overwrite:
        raise FileExistsError("Stage 14b outputs exist; pass --overwrite")

    raw_directory = outputs["raw_mmcif_directory"]
    aligned_directory = outputs["aligned_directory"]
    raw_paths = {
        row["pdb_id"]: (
            reference_path
            if row["pdb_id"] == "3KL6"
            else raw_directory / f"{row['pdb_id']}.cif"
        )
        for row in metadata_rows
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
    reference_structure = gemmi.read_structure(str(reference_path))
    reference_atoms = select_chain_atoms(reference_structure, str(reference["auth_chain"]))
    pocket_numbers = [int(value) for value in reference["pocket_residue_numbers"]]
    anchor_numbers = {int(value) for value in reference["anchor_residue_numbers"]}
    if derive_reference_residues(reference_atoms, str(reference["ligand_comp_id"]), 6.0) != pocket_numbers:
        raise ValueError("Stage 14b reference pocket reconstruction differs")
    if derive_reference_residues(reference_atoms, str(reference["ligand_comp_id"]), 4.0) != sorted(anchor_numbers):
        raise ValueError("Stage 14b reference anchor reconstruction differs")
    if len(ca_map(reference_atoms)) != int(reference["visible_protein_ca_count"]):
        raise ValueError("Stage 14b reference C-alpha count differs")
    reference_ligands = ligand_residue_map(reference_atoms, {str(reference["ligand_comp_id"])})
    if len(reference_ligands) != 1:
        raise ValueError("Stage 14b reference ligand is ambiguous")
    reference_ligand_coords = np.vstack(
        [atom.coord for atom in next(iter(reference_ligands.values()))]
    )

    gate = dict(config["coordinate_gate"])
    audit_rows: list[dict[str, object]] = []
    vectors: dict[str, np.ndarray] = {}
    feature_names: list[str] | None = None
    metadata_by_id = {row["pdb_id"]: row for row in metadata_rows}
    for pdb_id in sorted(metadata_by_id):
        metadata = metadata_by_id[pdb_id]
        conformer_id = fa10_conformer_id(pdb_id)
        raw_path = raw_paths[pdb_id]
        aligned_path = aligned_directory / f"{conformer_id}_to_3KL6_A.pdb"
        if pdb_id in errors:
            audit_rows.append(
                {
                    "conformer_id": conformer_id,
                    "pdb_id": pdb_id,
                    "chain": metadata["selected_auth_chain"],
                    "status": "download_failed",
                    "exclusion_reasons": "mmcif_download_failed",
                    "download_error": errors[pdb_id],
                }
            )
            continue
        qualifying_ids = {
            value for value in metadata["qualifying_ligand_ids"].split(";") if value
        }
        try:
            row, vector, names = audit_structure(
                pdb_id,
                metadata["selected_auth_chain"],
                qualifying_ids,
                raw_path,
                aligned_path,
                reference_atoms,
                reference_ligand_coords,
                pocket_numbers,
                anchor_numbers,
                gate,
            )
            row["conformer_id"] = conformer_id
            if row["status"] == "coordinate_eligible":
                normalize_fa10_alignment(aligned_path)
                row["aligned_protein_pdb_sha256"] = file_sha256(aligned_path)
            if (
                row["status"] == "coordinate_eligible"
                and int(row["global_incomplete_standard_amino_acid_residue_count"]) > 0
            ):
                row["status"] = "preparation_excluded"
                row["exclusion_reasons"] = "global_incomplete_standard_amino_acid_template"
                vector = None
                names = None
            row.update(
                {
                    "title": metadata["title"],
                    "resolution_angstrom": metadata["resolution_angstrom"],
                    "qualifying_ligand_ids": metadata["qualifying_ligand_ids"],
                }
            )
            audit_rows.append(row)
            if row["status"] == "coordinate_eligible":
                assert vector is not None and names is not None
                vectors[conformer_id] = vector
                if feature_names is None:
                    feature_names = names
                elif names != feature_names:
                    raise ValueError("Stage 14b structural feature names differ")
        except Exception as error:
            audit_rows.append(
                {
                    "conformer_id": conformer_id,
                    "pdb_id": pdb_id,
                    "chain": metadata["selected_auth_chain"],
                    "status": "coordinate_error",
                    "exclusion_reasons": "coordinate_audit_error",
                    "coordinate_error": f"{type(error).__name__}: {error}",
                    "mmcif_path": raw_path.relative_to(root).as_posix(),
                    "mmcif_sha256": file_sha256(raw_path),
                    "title": metadata["title"],
                    "resolution_angstrom": metadata["resolution_angstrom"],
                }
            )

    reference_id = "FA10_3KL6_reference"
    eligible_by_id = {
        str(row["conformer_id"]): row
        for row in audit_rows
        if row["status"] == "coordinate_eligible"
    }
    if reference_id not in eligible_by_id or feature_names is None:
        raise ValueError("Stage 14b reference failed structural eligibility")
    target_count = int(dict(preregistration["receptor_pool_plan"])["target_receptor_count"])
    reserve_count = int(dict(preregistration["receptor_pool_plan"])["reserve_receptor_count"])
    if len(eligible_by_id) < target_count + reserve_count:
        raise ValueError("too few Stage 14b preparation-ready candidates")

    ordered_ids = sorted(vectors)
    matrix = np.vstack([vectors[conformer_id] for conformer_id in ordered_ids])
    deviations = matrix.std(axis=0)
    keep = deviations >= float(dict(config["structural_selection"])["minimum_variable_feature_sd_angstrom"])
    variable_count = int(keep.sum())
    if variable_count < 3:
        raise ValueError("too few variable Stage 14b structural features")
    standardized = (matrix[:, keep] - matrix[:, keep].mean(axis=0)) / deviations[keep]
    standardized /= math.sqrt(variable_count)
    distances: dict[tuple[str, str], float] = {}
    distance_rows: list[dict[str, object]] = []
    for first_index, second_index in combinations(range(len(ordered_ids)), 2):
        first = ordered_ids[first_index]
        second = ordered_ids[second_index]
        distance = float(np.linalg.norm(standardized[first_index] - standardized[second_index]))
        distances[(first, second)] = distance
        distance_rows.append(
            {
                "conformer_id_a": first,
                "conformer_id_b": second,
                "standardized_active_site_distance": distance,
            }
        )
    additions = maxmin_select(
        ordered_ids,
        [reference_id],
        distances,
        target_count + reserve_count - 1,
    )
    ranked_ids = [reference_id] + [str(row["conformer_id"]) for row in additions]
    rank_metrics = {reference_id: ""}
    rank_metrics.update(
        {
            str(row["conformer_id"]): row["minimum_standardized_distance_to_selected_pool"]
            for row in additions
        }
    )
    selected_rows = [
        {
            "pool_role": "reference_seed" if rank == 1 else "maxmin_addition",
            "selection_rank": rank,
            "minimum_standardized_distance_to_selected_pool": rank_metrics[conformer_id],
            **eligible_by_id[conformer_id],
        }
        for rank, conformer_id in enumerate(ranked_ids[:target_count], start=1)
    ]
    reserve_rows = [
        {
            "pool_role": "frozen_reserve",
            "reserve_rank": rank,
            "global_selection_rank": target_count + rank,
            "minimum_standardized_distance_to_selected_pool": rank_metrics[conformer_id],
            **eligible_by_id[conformer_id],
        }
        for rank, conformer_id in enumerate(ranked_ids[target_count:], start=1)
    ]
    feature_rows = [
        {
            "conformer_id": conformer_id,
            **{name: float(value) for name, value in zip(feature_names, vectors[conformer_id])},
        }
        for conformer_id in ordered_ids
    ]
    eligible_rows = [eligible_by_id[conformer_id] for conformer_id in ordered_ids]

    write_csv(outputs["coordinate_audit_csv"], audit_rows)
    write_csv(outputs["eligible_pool_csv"], eligible_rows)
    write_csv(outputs["feature_matrix_csv"], feature_rows)
    write_csv(outputs["pairwise_distances_csv"], distance_rows)
    write_csv(outputs["selected16_csv"], selected_rows)
    write_csv(outputs["reserve_csv"], reserve_rows)
    reason_counts = Counter(
        reason
        for row in audit_rows
        for reason in str(row.get("exclusion_reasons", "")).split(";")
        if reason
    )
    summary = {
        "schema_version": "1.0",
        "experiment_id": config["experiment_id"],
        "status": "stage14b_fa10_structural_pool_ok",
        "config": {
            "path": config_path.relative_to(root).as_posix(),
            "sha256": file_sha256(config_path),
        },
        "runtime": runtime,
        "counts": {
            "audited_count": len(audit_rows),
            "download_failure_count": len(errors),
            "coordinate_eligible_count": sum(row["status"] == "coordinate_eligible" for row in audit_rows),
            "preparation_ready_count": len(eligible_rows),
            "selected_receptor_count": len(selected_rows),
            "reserve_receptor_count": len(reserve_rows),
            "raw_feature_count": len(feature_names),
            "variable_feature_count": variable_count,
            "pairwise_distance_count": len(distance_rows),
        },
        "exclusion_reason_counts": dict(sorted(reason_counts.items())),
        "reference": {
            "conformer_id": reference_id,
            "pocket_residue_numbers": pocket_numbers,
            "anchor_residue_numbers": sorted(anchor_numbers),
        },
        "selected_receptor_ids": [str(row["conformer_id"]) for row in selected_rows],
        "reserve_receptor_ids": [str(row["conformer_id"]) for row in reserve_rows],
        "data_boundary": {
            "ligand_labels_read": 0,
            "docking_scores_read": 0,
            "fresh_validation_rows_read": 0,
            "test_rows_read": 0,
        },
        "outputs": {
            key: {
                "path": path.relative_to(root).as_posix(),
                "sha256": file_sha256(path),
            }
            for key, path in outputs.items()
            if key not in {"raw_mmcif_directory", "aligned_directory", "summary_json", "report_md"}
        },
        "next_gate": "prepare the selected FA10 receptors and cognate ligands, derive one common box, and run three-seed Uni-Dock redocking",
        "interpretation_boundary": config["interpretation_boundary"],
    }
    write_json(outputs["summary_json"], summary)
    write_report(outputs["report_md"], summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
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
